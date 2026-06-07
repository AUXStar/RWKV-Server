import torch
import gc
import time
from loguru import logger

from .loader import RWKV070ModelLoader
from .scheduler_base import BaseScheduler
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
        super().__init__(model_loader, max_batch_size, buffer_size)

        cap = self._next_power_of_two(initial_capacity)
        self.current_capacity = cap if cap <= max_batch_size else max_batch_size
        self.mask = torch.ones(self.current_capacity, dtype=torch.bool, device="cuda")
        log.info(f"Init capacity = {self.current_capacity} (max={max_batch_size})")

    @staticmethod
    def _next_power_of_two(x: int) -> int:
        return 1 << (x - 1).bit_length() if x > 1 else 1

    def _get_active_indices(self):
        """获取当前活跃任务的索引列表"""
        return [
            i
            for i in range(self.current_capacity)
            if not self.mask[i].item() and self.worker["tasks"][i] is not None
        ]

    def _get_active_count(self):
        """获取当前活跃任务的数量"""
        return len(self._get_active_indices())

    def _compact_and_resize(self, new_capacity: int):
        new_capacity = min(new_capacity, self.max_batch_size)
        if new_capacity == self.current_capacity:
            return
        t0 = time.time()
        
        active_indices = self._get_active_indices()
        active_cnt = len(active_indices)
        w = self.worker
        state_size = 64  # 每个随机状态占用的元素个数

        # 处理rand_state（特殊逻辑，保留原实现）
        total_elements = w["rand_state"].numel()
        total_slots = total_elements // state_size
        rand_view = w["rand_state"].view(total_slots, state_size)
        rand_view[:active_cnt] = rand_view[active_indices]

        if new_capacity > total_slots:
            new_rand = torch.empty(
                (new_capacity, state_size),
                dtype=w["rand_state"].dtype,
                device=w["rand_state"].device,
            )
            new_rand[:total_slots] = rand_view[:total_slots]
            new_rand[total_slots:].zero_()
            w["rand_state"] = new_rand.reshape(-1)
        else:
            w["rand_state"] = rand_view[:new_capacity].reshape(-1)

        # 统一移动所有张量（合并原重复代码）
        tensor_moves = [
            ("last_tokens", lambda x: x),
            ("max_tokens", lambda x: x),
            ("generated_tokens", lambda x: x),
            ("shift_state", lambda x: (slice(None), slice(None), x, slice(None))),
            ("wkv_state", lambda x: (slice(None), x, ...)),
            ("elapsed_t", lambda x: x),
            ("penalties", lambda x: x),
            ("stop_flags", lambda x: x),
            ("presence_penalties", lambda x: x),
            ("repetition_penalties", lambda x: x),
            ("penalty_decays", lambda x: x),
            ("temperatures", lambda x: x),
            ("top_ps", lambda x: x),
            ("top_ks", lambda x: x),
        ]
        
        for name, idx_fn in tensor_moves:
            w[name][idx_fn(slice(active_cnt))] = w[name][idx_fn(active_indices)]

        # 移动任务对象
        old_tasks = w["tasks"]
        new_tasks = [None] * new_capacity
        for i, idx in enumerate(active_indices):
            if i < new_capacity:
                new_tasks[i] = old_tasks[idx]
        w["tasks"] = new_tasks

        # 更新mask
        new_mask = torch.ones(new_capacity, dtype=torch.bool, device="cuda")
        new_mask[:active_cnt] = False
        self.mask = new_mask

        # 统一清零新增空闲槽位（合并原重复代码）
        if active_cnt < new_capacity:
            slice_range = slice(active_cnt, new_capacity)
            zero_tensors = [
                ("shift_state", lambda s: (slice(None), slice(None), s, slice(None))),
                ("wkv_state", lambda s: (slice(None), s, ...)),
                ("elapsed_t", lambda s: s),
                ("penalties", lambda s: s),
                ("stop_flags", lambda s: s),
            ]
            
            for name, idx_fn in zero_tensors:
                w[name][idx_fn(slice_range)].zero_()

        log.info(
            f"Compact {active_cnt} tasks, cap {self.current_capacity} -> {new_capacity} in {time.time()-t0:.4f}s"
        )
        self.current_capacity = new_capacity
        gc.collect()
        torch.cuda.empty_cache()

    def _adjust_capacity(self, ready_count: int = None):
        if ready_count is None:
            ready_count = sum(1 for t in self.tasks if t.status == Status.READY)
        active = self._get_active_count()
        total_needed = active + ready_count
        target = self._next_power_of_two(total_needed)
        if target > self.max_batch_size:
            target = self.max_batch_size
        if target != self.current_capacity:
            log.info(
                f"Adjust capacity {self.current_capacity} -> {target} (active={active}, ready={ready_count})"
            )
            self._compact_and_resize(target)

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

        # 统一字段映射（合并原重复赋值代码）
        field_mappings = [
            ("stop_flags", lambda s: s, False),
            ("last_tokens", lambda s: s, lambda t: t.current_token),
            ("max_tokens", lambda s: s, lambda t: t.max_tokens),
            ("tasks", lambda s: s, lambda t: t),
            ("shift_state", lambda s: (slice(None), slice(None), [s], slice(None)), lambda t: t.shift_state),
            ("wkv_state", lambda s: (slice(None), [s], ...), lambda t: t.wkv_state),
            ("elapsed_t", lambda s: s, lambda t: t.elapsed_t),
            ("penalties", lambda s: s, lambda t: t.penalties),
            ("rand_state", lambda s: slice(s * 64, (s + 1) * 64), lambda t: t.rand_state),
            ("presence_penalties", lambda s: s, lambda t: t.presence_penalty),
            ("repetition_penalties", lambda s: s, lambda t: t.repetition_penalty),
            ("penalty_decays", lambda s: s, lambda t: t.penalty_decay),
            ("temperatures", lambda s: s, lambda t: t.temperature),
            ("top_ps", lambda s: s, lambda t: t.top_p),
            ("top_ks", lambda s: s, lambda t: t.top_k),
        ]

        for i in range(num):
            slot = free_slots[i]
            task = ready_tasks[i]
            task.status = Status.RUNNING
            with task:  # 自动完成to gpu 然后to cpu
                task.stop_flag_tensor = w["stop_flags"][slot]
                # 统一处理所有字段赋值
                for field_name, idx_fn, value_fn in field_mappings:
                    idx = idx_fn(slot)
                    value = value_fn(task) if callable(value_fn) else value_fn
                    w[field_name][idx] = value
                mask[slot] = False

    def _process_terminated_indices(self, indices, data_generator=None):
        """
        处理真正终止的任务（zero_row和partial_row）
        :param indices: 终止的任务索引张量
        :param data_generator: 生成每个任务收集数据的函数
        """
        if indices.numel() == 0:
            return
        indices_list = indices.tolist()
        tasks = self.worker["tasks"][:self.current_capacity]
        
        for idx, i in enumerate(indices_list):
            task = tasks[i]
            if task is None:
                continue
            # 生成要收集的token数据
            tokens = data_generator(idx, i) if data_generator else []
            task.collect(tokens)
            task.status = Status.FINISHED
            task.finish()
            tasks[i] = None
            self.worker["tasks"][i] = None
        
        # 统一更新mask和stop_flags
        self.mask[indices] = True
        self.worker["stop_flags"][indices] = False

    def _process_full_buffer_indices(self, indices, data_generator):
        """
        处理满buffer生成的任务（all_row）：只收集数据，不终止任务
        :param indices: 满buffer的任务索引张量
        :param data_generator: 生成每个任务收集数据的函数
        """
        if indices.numel() == 0:
            return
        indices_list = indices.tolist()
        tasks = self.worker["tasks"][:self.current_capacity]
        
        for idx, i in enumerate(indices_list):
            task = tasks[i]
            if task is None:
                continue
            # 只收集数据，不修改任务状态
            tokens = data_generator(idx, i)
            task.collect(tokens)
        
        # 只重置stop_flags，不修改mask
        self.worker["stop_flags"][indices] = False

    def run(self):
        self._adjust_capacity(
            ready_count=sum(1 for t in self.tasks if t.status == Status.READY)
        )
        it = 0
        while self.tasks:
            it += 1
            self.update_batch(self.mask)
            self._adjust_capacity()
            self.update_batch(self.mask)

            active_cnt = self._get_active_count()
            if active_cnt == 0 and not any(
                t.status == Status.READY for t in self.tasks
            ):
                break

            st = time.time()
            # 传入 stop_flags
            self.engine.generate(self.worker, self.mask, self.worker["stop_flags"])
            pulse_time = time.time() - st
            self._collect()

            active = self._get_active_count()
            if active and pulse_time > 0:
                speed = active * self.buffer_size / pulse_time
                log.info(
                    f"Iter {it} cap={self.current_capacity} active={active} speed={speed:.2f} tok/s per_task={speed/active:.2f} tok/s"
                )

    def _collect(self):
        t0 = time.time()
        cur = self.current_capacity
        gen = self.worker["generated_tokens"][:cur, : self.buffer_size]
        tasks = self.worker["tasks"][:cur]
        nonz = gen != 0
        any_row = nonz.any(dim=1)
        all_row = nonz.all(dim=1)
        zero_row = ~any_row

        self._process_terminated_indices(torch.where(zero_row)[0].cpu())

        all_indices = torch.where(all_row)[0].cpu()
        gen_all = gen[all_indices].cpu()
        self._process_full_buffer_indices(
            all_indices,
            data_generator=lambda idx, _: gen_all[idx].tolist()
        )

        partial_indices = torch.where(any_row & ~all_row)[0].cpu()
        gen_partial = gen[partial_indices].cpu()
        
        def partial_data_gen(idx, i):
            row = gen_partial[idx]
            val = row[row != 0].tolist()
            if tasks[i].current_token == 0:
                val.append(0)
            return val
        
        self._process_terminated_indices(partial_indices, data_generator=partial_data_gen)

        with self._tasks_lock:
            self.tasks = [t for t in self.tasks if t.status != Status.FINISHED]
        log.info(f"Collect finished in {time.time()-t0:.2f}s")