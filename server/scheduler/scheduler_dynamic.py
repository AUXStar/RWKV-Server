import torch
import gc
import time

from .loader import RWKV070ModelLoader
from .scheduler_base import BaseScheduler
from ..task.task import Status

initial_capacity: int = 1


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
        # 保留一个切片，减小占用，避免多次切取
        self.mask = self.worker["mask"][: self.current_capacity]

        self.log.info(f"Init capacity = {self.current_capacity} (max={max_batch_size})")

    @staticmethod
    def _next_power_of_two(x: int) -> int:
        return 1 << (x - 1).bit_length() if x > 1 else 1

    def _compact_and_resize(self, new_capacity: int):
        new_capacity = min(new_capacity, self.max_batch_size)
        if new_capacity == self.current_capacity:
            return
        t0 = time.time()

        active_indices = self._get_active_indices()
        active_cnt = len(active_indices)
        w = self.worker
        state_size = 64  # 每个随机状态占用的元素个数

        # 处理rand_state
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

        # 统一移动所有张量
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
        w["mask"].fill_(True)
        self.mask = w["mask"][:new_capacity]
        self.mask[:active_cnt] = False

        # 统一清零新增空闲槽位
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

        self.log.info(
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
            self.log.info(
                f"Adjust capacity {self.current_capacity} -> {target} (active={active}, ready={ready_count})"
            )
            self._compact_and_resize(target)

    def run(self):
        self._adjust_capacity(
            ready_count=sum(1 for t in self.tasks if t.status == Status.READY)
        )
        it = 0
        while self.tasks:
            it += 1
            self.update_batch()
            self._adjust_capacity()
            self.update_batch()

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

            self.log_speed(it,pulse_time)
