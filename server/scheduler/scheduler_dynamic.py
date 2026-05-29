import torch
import gc
import time
from loguru import logger

from .loader import RWKV070ModelLoader
from .scheduler_base import BaseScheduler
from .sampler import BatchSampler
from ..task import Status
from .batch_engine import InferEngine

initial_capacity: int = 1
log = logger.bind(module="scheduler.dynamic")


class DynamicScheduler(BaseScheduler):
    def __init__(
        self,
        model_loader: RWKV070ModelLoader,
        max_batch_size: int = 35,
        buffer_size: int = 50,
    ):
        super().__init__(model_loader, BatchSampler, max_batch_size, buffer_size)
        self.worker = self._init_worker_slots(self.max_batch_size)
        self.worker["stop_flags"] = torch.zeros(
            self.max_batch_size, dtype=torch.bool, device="cuda"
        )
        self.engine = InferEngine(
            self.model_loader.model, self.sampler, self.buffer_size, self.model_loader.batch_is_eos
        )
        cap = self._next_power_of_two(initial_capacity)
        self.current_capacity = cap if cap <= max_batch_size else max_batch_size
        self.mask = torch.ones(self.current_capacity, dtype=torch.bool, device="cuda")
        log.info(f"Init capacity = {self.current_capacity} (max={max_batch_size})")

    @staticmethod
    def _next_power_of_two(x: int) -> int:
        return 1 << (x - 1).bit_length() if x > 1 else 1

    def _compact_and_resize(self, new_capacity: int):
        new_capacity = min(new_capacity, self.max_batch_size)
        if new_capacity == self.current_capacity:
            return
        t0 = time.time()
        active_indices = [
            i
            for i in range(self.current_capacity)
            if not self.mask[i].item() and self.worker["tasks"][i] is not None
        ]
        active_cnt = len(active_indices)
        w = self.worker
        state_size = 64  # 每个随机状态占用的元素个数

        # 获取当前 rand_state 的物理槽位数（基于实际元素个数）
        total_elements = w["rand_state"].numel()
        total_slots = total_elements // state_size
        # 断言 active_indices 中的最大值 < total_slots
        rand_view = w["rand_state"].view(total_slots, state_size)

        # 移动数据（注意 active_indices 是原始槽位索引，必须 < total_slots）
        # 只移动前 active_cnt 个位置
        rand_view[:active_cnt] = rand_view[active_indices]

        # 创建新视图（物理容量不变，但逻辑视图缩小至 new_capacity）
        # 如果 new_capacity > total_slots，需要重新分配更大的张量（一般不会发生，因为 new_capacity <= max_batch_size 且初始化时 total_slots >= max_batch_size）
        if new_capacity > total_slots:
            # 扩容：需要分配更大的 rand_state，复制旧数据后再添加新空间
            new_rand = torch.empty(
                (new_capacity, state_size),
                dtype=w["rand_state"].dtype,
                device=w["rand_state"].device,
            )
            new_rand[:total_slots] = rand_view[:total_slots]
            if new_capacity > total_slots:
                new_rand[total_slots:].zero_()
            w["rand_state"] = new_rand.reshape(-1)
        else:
            # 缩小：直接切片视图
            w["rand_state"] = rand_view[:new_capacity].reshape(-1)

        # 移动其他张量（原有逻辑不变）
        w["last_tokens"][:active_cnt] = w["last_tokens"][active_indices]
        w["max_tokens"][:active_cnt] = w["max_tokens"][active_indices]
        w["generated_tokens"][:active_cnt] = w["generated_tokens"][active_indices]
        w["shift_state"][:, :, :active_cnt, :] = w["shift_state"][:, :, active_indices, :]
        w["wkv_state"][:, :active_cnt, ...] = w["wkv_state"][:, active_indices, ...]
        w["elapsed_t"][:active_cnt] = w["elapsed_t"][active_indices]
        w["penalties"][:active_cnt] = w["penalties"][active_indices]
        w["stop_flags"][:active_cnt] = w["stop_flags"][active_indices]

        for k in [
            "presence_penalties",
            "repetition_penalties",
            "penalty_decays",
            "temperatures",
            "top_ps",
            "top_ks",
        ]:
            w[k][:active_cnt] = w[k][active_indices]

        # 移动任务对象
        old_tasks = w["tasks"]
        new_tasks = [None] * new_capacity
        for i, idx in enumerate(active_indices):
            if i < new_capacity:
                new_tasks[i] = old_tasks[idx]
        w["tasks"] = new_tasks

        # 更新 mask
        new_mask = torch.ones(new_capacity, dtype=torch.bool, device="cuda")
        new_mask[:active_cnt] = False
        self.mask = new_mask

        # 清理新增空闲槽位（如果 new_capacity > active_cnt）
        if active_cnt < new_capacity:
            w["shift_state"][:, :, active_cnt:new_capacity, :].zero_()
            w["wkv_state"][:, active_cnt:new_capacity, ...].zero_()
            w["elapsed_t"][active_cnt:new_capacity].zero_()
            w["penalties"][active_cnt:new_capacity].zero_()
            w["stop_flags"][active_cnt:new_capacity].zero_()

        gc.collect()
        torch.cuda.empty_cache()
        log.info(
            f"Compact {active_cnt} tasks, cap {self.current_capacity} -> {new_capacity} in {time.time()-t0:.4f}s"
        )
        self.current_capacity = new_capacity

    def _adjust_capacity(self, ready_count: int = None):
        if ready_count is None:
            ready_count = sum(1 for t in self.tasks if t.status == Status.READY)
        active = sum(
            1
            for i in range(self.current_capacity)
            if not self.mask[i].item() and self.worker["tasks"][i] is not None
        )
        total_needed = active + ready_count
        target = self._next_power_of_two(total_needed)
        if target > self.max_batch_size:
            target = self.max_batch_size
        if target != self.current_capacity:
            log.info(
                f"Adjust capacity {self.current_capacity} -> {target} (active={active}, ready={ready_count})"
            )
            self._compact_and_resize(target)
        gc.collect()
        torch.cuda.empty_cache()

    def clear_worker(self):
        self._clear_worker(self.worker)

    def update_batch(self, mask: torch.Tensor):
        free_slots = [
            i
            for i in range(self.current_capacity)
            if mask[i].item() and self.worker["tasks"][i] is None
        ]
        if not free_slots:
            return
        ready_tasks = [t for t in self.tasks if t.status == Status.READY]
        if not ready_tasks:
            return
        num = min(len(free_slots), len(ready_tasks))
        w = self.worker
        gc.collect()
        torch.cuda.empty_cache()
        for i in range(num):
            slot = free_slots[i]
            task = ready_tasks[i]
            task.status = Status.RUNNING
            with task: # 自动完成to gpu 然后to cpu
                task.stop_flag_tensor = w["stop_flags"][slot]
                w["stop_flags"][slot] = False
                w["last_tokens"][slot] = task.current_token
                w["max_tokens"][slot] = task.max_tokens
                w["tasks"][slot] = task
                w["shift_state"][:, :, [slot], :] = task.shift_state
                w["wkv_state"][:, [slot], ...] = task.wkv_state
                w["elapsed_t"][slot] = task.elapsed_t
                w["penalties"][slot] = task.penalties
                w["rand_state"][slot * 64 : (slot + 1) * 64] = task.rand_state
                for k, v in [
                    ("presence_penalties", task.presence_penalty),
                    ("repetition_penalties", task.repetition_penalty),
                    ("penalty_decays", task.penalty_decay),
                    ("temperatures", task.temperature),
                    ("top_ps", task.top_p),
                    ("top_ks", task.top_k),
                ]:
                    w[k][slot] = v
                mask[slot] = False

    def run(self):
        self._adjust_capacity(
            ready_count=sum(1 for t in self.tasks if t.status == Status.READY)
        )
        it = 0
        while self.tasks:
            gc.collect()
            torch.cuda.empty_cache()
            it += 1
            self.update_batch(self.mask)
            self._adjust_capacity()
            self.update_batch(self.mask)

            active_cnt = sum(
                1
                for i in range(self.current_capacity)
                if not self.mask[i].item() and self.worker["tasks"][i] is not None
            )
            if active_cnt == 0 and not any(
                t.status == Status.READY for t in self.tasks
            ):
                break

            st = time.time()
            # 传入 stop_flags
            self.engine.generate(self.worker, self.mask, self.worker["stop_flags"])
            pulse_time = time.time() - st
            self._collect()

            active = sum(
                1
                for i in range(self.current_capacity)
                if not self.mask[i].item() and self.worker["tasks"][i] is not None
            )
            if active and pulse_time > 0:
                speed = active * self.buffer_size / pulse_time
                log.info(
                    f"Iter {it} cap={self.current_capacity} active={active} speed={speed:.2f} tok/s"
                )

    def _collect(self):
        t0 = time.time()
        cur = self.current_capacity
        gen = self.worker["generated_tokens"][:cur, : self.buffer_size]
        tasks = self.worker["tasks"][:cur]
        nonz = gen != 0
        any_row = nonz.any(dim=1)
        all_row = nonz.all(dim=1)
        for i in torch.where(all_row)[0].cpu().numpy():
            if tasks[i] is None:
                continue
            tasks[i].collect(gen[i].tolist())
            self.worker["stop_flags"][i] = False
        for i in torch.where(any_row & ~all_row)[0].cpu().numpy():
            if tasks[i] is None:
                continue
            val = gen[i][gen[i] != 0].tolist()
            if tasks[i].current_token == 0:
                val.append(0)
            tasks[i].status = Status.FINISHED
            tasks[i].collect(val)
            tasks[i].finish()
            tasks[i] = None
            self.mask[i] = True
            self.worker["tasks"][i] = None
            self.worker["stop_flags"][i] = False
        self.tasks = [t for t in self.tasks if t.status != Status.FINISHED]
        log.info(f"Collect finished in {time.time()-t0:.2f}s")
