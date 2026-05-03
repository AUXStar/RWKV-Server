import torch
import time
from loguru import logger

from server.pipeline.loader import RWKV070ModelLoader
from server.pipeline.manager_base import TaskManagerBase, WorkerState
from server.pipeline.sampler import BatchSampler
from server.pipeline.task import Status, Task

initial_capacity: int = 1
log = logger.bind(module="taskmanager.dynamic")


class DynamicTaskManager(TaskManagerBase):
    def __init__(
        self,
        model_loader: RWKV070ModelLoader,
        max_batch_size: int = 15,
        buffer_size: int = 50,
    ):
        super().__init__(model_loader, BatchSampler, max_batch_size, buffer_size)

        self.worker = self._init_worker_slots(self.max_batch_size)
        # 添加 stop_flags 张量
        self.worker["stop_flags"] = torch.zeros(self.max_batch_size, dtype=torch.bool, device="cuda")

        self.max_batch_size = max_batch_size
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
        cap_max = self.max_batch_size

        # 移动所有张量（包括 stop_flags）
        w["last_tokens"][:active_cnt] = w["last_tokens"][active_indices]
        w["max_tokens"][:active_cnt] = w["max_tokens"][active_indices]
        w["generated_tokens"][:active_cnt] = w["generated_tokens"][active_indices]
        w["state"][0][:, :, :active_cnt, :] = w["state"][0][:, :, active_indices, :]
        w["state"][1][:, :active_cnt, ...] = w["state"][1][:, active_indices, ...]
        w["penalties"][:active_cnt] = w["penalties"][active_indices]
        w["stop_flags"][:active_cnt] = w["stop_flags"][active_indices]   # 新增

        rand_view = w["rand_state"].view(cap_max, -1)
        rand_view[:active_cnt] = rand_view[active_indices]
        w["rand_state"] = rand_view.reshape(-1)

        for k in [
            "presence_penalties",
            "repetition_penalties",
            "penalty_decays",
            "temperatures",
            "top_ps",
            "top_ks",
        ]:
            w[k][:active_cnt] = w[k][active_indices]

        old_tasks = w["tasks"]
        new_tasks = [None] * new_capacity
        for i, idx in enumerate(active_indices):
            if i < new_capacity:
                new_tasks[i] = old_tasks[idx]
        w["tasks"] = new_tasks

        new_mask = torch.ones(new_capacity, dtype=torch.bool, device="cuda")
        new_mask[:active_cnt] = False
        self.mask = new_mask
        old_cap = self.current_capacity
        self.current_capacity = new_capacity

        if active_cnt < new_capacity:
            w["state"][0][:, :, active_cnt:new_capacity, :].zero_()
            w["state"][1][:, active_cnt:new_capacity, ...].zero_()
            w["penalties"][active_cnt:new_capacity].zero_()
            w["stop_flags"][active_cnt:new_capacity].zero_()   # 新增

        log.info(
            f"Compact {active_cnt} tasks, cap {old_cap} -> {new_capacity} (done in {time.time()-t0:.4f}s)"
        )

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
        for i in range(num):
            slot = free_slots[i]
            task = ready_tasks[i]
            task.status = Status.RUNNING
            task.prepare()
            # 绑定 stop_flag_tensor
            task.stop_flag_tensor = w["stop_flags"][slot]   # 标量视图
            # 确保初始为 False
            w["stop_flags"][slot] = False

            w["last_tokens"][slot] = task.current_token
            w["max_tokens"][slot] = task.max_tokens
            w["tasks"][slot] = task
            w["state"][0][:, :, slot, :] = task.state0
            w["state"][1][:, slot, ...] = task.state1
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
            self._task_single(self.mask)
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

    @torch.no_grad()
    def _task_single(self, raw_mask):
        t0 = time.time()
        w = self.worker
        cur = self.current_capacity
        mask = raw_mask.clone()

        last = w["last_tokens"][:cur]
        max_tok = w["max_tokens"][:cur]
        gen = w["generated_tokens"][:cur]
        pen = w["penalties"][:cur]
        rand = w["rand_state"][: cur * 64]

        p_pen = w["presence_penalties"][:cur]
        r_pen = w["repetition_penalties"][:cur]
        p_decay = w["penalty_decays"][:cur]
        temp = w["temperatures"][:cur]
        top_p = w["top_ps"][:cur]
        top_k = w["top_ks"][:cur]

        s0_full = w["state"][0]
        s1_full = w["state"][1]
        stop_flags = w["stop_flags"][:cur]   # GPU 张量

        s0_full[:, :, :cur, :] *= ~mask.view(1, 1, cur, 1)
        s1_full[:, :cur, ...] *= ~mask.view(1, cur, 1, 1, 1)

        step = 0
        for step in range(self.buffer_size):
            s0 = s0_full[:, :, :cur, :]
            s1 = s1_full[:, :cur, ...]
            logits = self.model.model.patch_forward_seq_batch(
                last.unsqueeze(-1), [s0, s1]
            )

            tokens = self.sampler.sample(
                logits=logits,
                penalties=pen,
                states=rand,
                presence_penalties=p_pen,
                repetition_penalties=r_pen,
                penalty_decays=p_decay,
                temperatures=temp,
                top_ps=top_p,
                top_ks=top_k,
                eos_mask=mask,
            )

            gen[mask, step] = 0
            eos1 = tokens == 0
            eos2 = (last == 261) & (tokens == 24281)
            finish = eos1 | eos2
            tokens[eos2] = 0

            # 停止标志：活跃且 stop_flags 为 True
            stop_mask = stop_flags & ~mask

            max_tok[~mask] -= 1
            # 增加 stop_mask 条件
            new_fin = (finish | (max_tok <= 0) | stop_mask) & ~mask
            gen[~mask, step] = tokens[~mask]

            if new_fin.any():
                s0_full[:, :, :cur, :] *= ~new_fin.view(1, 1, cur, 1)
                s1_full[:, :cur, ...] *= ~new_fin.view(1, cur, 1, 1, 1)
                for idx in torch.where(new_fin)[0].tolist():
                    task = w["tasks"][idx]
                    if task is None:
                        continue
                    task.current_token = tokens[idx].item()
                    task.state0.copy_(s0_full[:, :, idx])
                    task.state1.copy_(s1_full[:, idx])
                    task.penalties.copy_(w["penalties"][idx])
                    task.rand_state.copy_(w["rand_state"][idx * 64 : (idx + 1) * 64])
                    # 重置 stop_flag（如果因 stop 而结束，以免影响后续复用）
                    stop_flags[idx] = False

            mask |= new_fin
            tokens[mask] = 0
            last.copy_(tokens)
            if mask.all():
                gen[mask, step + 1 :] = 0
                break

        raw_mask.copy_(mask)
        log.info(f"Pulse {step+1} steps in {time.time()-t0:.2f}s")

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
            self.worker["stop_flags"][i] = False   # 重置
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
            self.worker["stop_flags"][i] = False   # 重置

        self.tasks = [t for t in self.tasks if t.status != Status.FINISHED]
        log.info(f"Collect finished in {time.time()-t0:.2f}s")