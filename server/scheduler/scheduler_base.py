from abc import ABC, abstractmethod
from typing import List, Optional
from loguru import logger
import torch
import threading
import time

from .loader import RWKV070ModelLoader
from .batch_engine import InferEngine
from .batch_sampler import BatchSampler
from ..task.task import Task, Status


class BaseScheduler(ABC):
    per_speed:int = 0
    def __init__(
        self,
        model_loader: RWKV070ModelLoader,
        max_batch_size: int,
        buffer_size: int = 32,
    ):
        self.model_loader = model_loader
        self.sampler = BatchSampler()
        self.max_batch_size = max_batch_size
        # 默认初始能力max cap，适应不同规划方法
        self.current_capacity = max_batch_size

        self.buffer_size = buffer_size
        self.tasks: List[Task] = []
        self.finished_tasks: List[Task] = []
        self._stop_event = False
        self._tasks_lock = threading.Lock()

        self.worker = self._init_worker_slots(self.max_batch_size)

        self.engine = InferEngine(
            self.model_loader.model,
            self.sampler,
            self.buffer_size,
            self.model_loader.batch_is_eos,
        )
        self.mask = self.worker["mask"]
        module_name = f'scheduler.{self.__class__.__name__.replace("Scheduler", "").lower()}'
        self.log = logger.bind(module=module_name)
        self.log.info(f"Init {module_name} with max batch size {self.max_batch_size}")

    def new_task(
        self,
        prompt: str | list[int],
        max_tokens: int = 50,
        repetition_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_decay: float = 0.0,
        temperature: float = 0.3,
        top_p: float = 0.1,
        top_k: int = 20,
        seed: int = 42,
        collect_callback: Optional[callable] = None,
        finish_callback: Optional[callable] = None,
    ) -> Task:
        task = Task(
            prompt=prompt,
            model_loader=self.model_loader,
            batch_sampler=self.sampler,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            penalty_decay=penalty_decay,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            collect_callback=collect_callback,
            finish_callback=finish_callback,
        )
        with self._tasks_lock:
            self.tasks.append(task)
        return task

    def _init_worker_slots(self, batch_size: int) -> dict:
        vocab_size = self.model_loader.vocab_size
        shift_state, wkv_state, elapsed_t = self.model_loader.gen_state(batch_size)
        # torch.Size([32, 2, B, 2560]) torch.Size([32, B, 40, 64, 64]) torch.Size([B])
        return {
            "last_tokens": torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
            "max_tokens": torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
            "tasks": [None] * batch_size,
            "generated_tokens": torch.zeros(
                (batch_size, self.buffer_size), dtype=torch.int32, device="cuda"
            ),
            "shift_state": shift_state,
            "wkv_state": wkv_state,
            "elapsed_t": elapsed_t,
            "penalties": torch.zeros(
                batch_size, vocab_size, dtype=torch.float32, device="cuda"
            ),
            "rand_state": torch.zeros(64 * batch_size, dtype=torch.int8, device="cuda"),
            "presence_penalties": torch.zeros(
                batch_size, dtype=torch.float32, device="cuda"
            ),
            "repetition_penalties": torch.zeros(
                batch_size, dtype=torch.float32, device="cuda"
            ),
            "penalty_decays": torch.zeros(
                batch_size, dtype=torch.float32, device="cuda"
            ),
            "temperatures": torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "top_ps": torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "top_ks": torch.zeros(batch_size, dtype=torch.int, device="cuda"),
            "stop_flags": torch.zeros(batch_size, dtype=torch.bool, device="cuda"),
            "mask": torch.ones(batch_size, dtype=torch.bool, device="cuda"),
        }

    def add_task(self, task: Task):
        """外部添加任务（已创建好的 Task 对象）"""
        with self._tasks_lock:
            self.tasks.append(task)

    def background(self):
        while 1:
            if self._stop_event:
                return
            if self.tasks:
                try:
                    self.run()
                except Exception as e:
                    self.log.exception(e)

            else:
                time.sleep(0.5)

    def start_daemon(self):
        self.backthr = threading.Thread(target=self.background, daemon=True)
        self.backthr.start()
        return self.backthr

    def shutdown(self):
        [t.stop() for t in self.tasks]
        self.log.success("等待任务完成回收-5s")
        time.sleep(5)
        if not self._stop_event:
            self._stop_event = True

    def run(self):
        it = 0
        while self.tasks:
            it += 1
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

    def log_speed(self,it,pulse_time):
        active = self._get_active_count()
        if active and pulse_time > 0:
            speed = active * self.buffer_size / pulse_time
            self.per_speed = speed/active
            self.log.info(
                f"Iter {it} cap={self.current_capacity} active={active} speed={speed:.2f} tok/s per_task={self.per_speed:.2f} tok/s"
            )

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

    def update_batch(self):
        free_slots = [
            i
            for i in range(self.current_capacity)
            if self.mask[i].item() and self.worker["tasks"][i] is None
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
            (
                "shift_state",
                lambda s: (slice(None), slice(None), [s], slice(None)),
                lambda t: t.shift_state,
            ),
            ("wkv_state", lambda s: (slice(None), [s], ...), lambda t: t.wkv_state),
            ("elapsed_t", lambda s: s, lambda t: t.elapsed_t),
            ("penalties", lambda s: s, lambda t: t.penalties),
            (
                "rand_state",
                lambda s: slice(s * 64, (s + 1) * 64),
                lambda t: t.rand_state,
            ),
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
                self.mask[slot] = False

    def _process_terminated_indices(self, indices, data_generator=None):
        """
        处理真正终止的任务（zero_row和partial_row）
        :param indices: 终止的任务索引张量
        :param data_generator: 生成每个任务收集数据的函数
        """
        if indices.numel() == 0:
            return
        indices_list = indices.tolist()
        tasks = self.worker["tasks"][: self.current_capacity]

        for idx, i in enumerate(indices_list):
            task = tasks[i]
            if task is None:
                continue
            # 生成要收集的token数据
            tokens = data_generator(idx, i) if data_generator else []
            task.status = Status.FINISHED
            task.collect(tokens)
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
        tasks = self.worker["tasks"][: self.current_capacity]

        for idx, i in enumerate(indices_list):
            task = tasks[i]
            if task is None:
                continue
            # 只收集数据，不修改任务状态
            tokens = data_generator(idx, i)
            task.collect(tokens)

        # 只重置stop_flags，不修改mask
        self.worker["stop_flags"][indices] = False

    def _collect(self):
        t0 = time.time()
        cur = self.current_capacity
        gen = self.worker["generated_tokens"][:cur, : self.buffer_size]
        tasks = self.worker["tasks"][:cur]
        mask = self.mask[:cur]

        # 1. 优先处理在 generate() 中已经被 mask 的槽位（max_tok<=0 或 eos）
        masked_indices = torch.where(mask)[0].cpu()
        self._process_terminated_indices(
            masked_indices,
            data_generator=lambda idx, i: gen[i][gen[i] != 0].tolist()
        )

        # 2. 处理剩余未被 mask 的槽位
        remaining = ~mask
        non_z = (gen != 0) & remaining.unsqueeze(1)
        any_row = non_z.any(dim=1)
        all_row = non_z.all(dim=1)
        zero_row = ~any_row & remaining

        self._process_terminated_indices(torch.where(zero_row)[0].cpu())

        all_indices = torch.where(all_row)[0].cpu()
        gen_all = gen[all_indices].cpu()
        self._process_full_buffer_indices(
            all_indices, data_generator=lambda idx, _: gen_all[idx].tolist()
        )

        partial_indices = torch.where(any_row & ~all_row)[0].cpu()
        gen_partial = gen[partial_indices].cpu()

        def partial_data_gen(idx, i):
            row = gen_partial[idx]
            val = row[row != 0].tolist()
            if tasks[i].current_token == 0:
                val.append(0)
            return val

        self._process_terminated_indices(
            partial_indices, data_generator=partial_data_gen
        )

        with self._tasks_lock:
            self.tasks = [t for t in self.tasks if t.status != Status.FINISHED]
        self.log.info(f"Collect finished in {time.time()-t0:.2f}s")
