import torch
import time
from loguru import logger

from .loader import RWKV070ModelLoader
from .manager_base import BaseScheduler
from .sampler import BatchSampler
from .task import Status
from .batch_engine import InferEngine

log = logger.bind(module="scheduler.simple")


class SimpleScheduler(BaseScheduler):
    def __init__(
        self,
        model_loader: RWKV070ModelLoader,
        max_batch_size: int = 15,
        buffer_size: int = 100,
    ):
        super().__init__(model_loader, BatchSampler, max_batch_size, buffer_size)
        self.worker = self._init_worker_slots(self.max_batch_size)
        self.engine = InferEngine(
            self.model_loader.model,
            self.sampler,
            self.buffer_size,
            self.model_loader.batch_is_eos
        )
        log.info(f"Init capacity = {self.max_batch_size} (fixed)")

    def clear_worker(self):
        self._clear_worker(self.worker)

    def update_batch(self, mask: torch.Tensor):
        free_slots = torch.where(mask)[0].tolist()
        batch_tasks = [t for t in self.tasks if t.status == Status.READY]
        num_assign = min(len(free_slots), len(batch_tasks))
        w = self.worker
        for i in range(num_assign):
            slot = free_slots[i]
            task = batch_tasks[i]
            task.status = Status.RUNNING
            task.prepare()
            w["last_tokens"][slot] = task.current_token
            w["max_tokens"][slot] = task.max_tokens
            w["tasks"][slot] = task
            w["shift_state"][:, :, slot, :] = task.shift_state
            w["wkv_state"][:, slot, :, :, :] = task.wkv_state
            w["elapsed_t"][slot] = task.elapsed_t
            w["penalties"][slot] = task.penalties
            w["rand_state"][slot * 64 : (slot + 1) * 64] = task.rand_state
            w["presence_penalties"][slot] = task.presence_penalty
            w["repetition_penalties"][slot] = task.repetition_penalty
            w["penalty_decays"][slot] = task.penalty_decay
            w["temperatures"][slot] = task.temperature
            w["top_ps"][slot] = task.top_p
            w["top_ks"][slot] = task.top_k
            mask[slot] = False

    def run(self):
        mask = torch.ones(self.max_batch_size, dtype=torch.bool, device="cuda")
        it = 0
        while self.tasks:
            it += 1
            self.update_batch(mask)
            if mask.all():
                break
            st = time.time()
            self.engine.generate(self.worker, mask)
            pulse_time = time.time() - st
            self._collect()
            active = (~mask).sum().item()
            speed = active * self.buffer_size / pulse_time if active else 0
            log.info(f"Iter {it} active={active} speed={speed:.2f} tok/s")

    def _collect(self):
        t0 = time.time()
        gen = self.worker["generated_tokens"][:, :self.buffer_size]
        tasks = self.worker["tasks"]
        nonz = gen != 0
        row_any = nonz.any(dim=1)
        row_all = nonz.all(dim=1)
        for i in torch.where(row_all)[0].cpu().numpy():
            tasks[i].collect(gen[i].tolist())
            tasks[i].status = Status.FINISHED
            tasks[i].finish()
        for i in torch.where(row_any & ~row_all)[0].cpu().numpy():
            val = gen[i][gen[i] != 0].tolist()
            if tasks[i].current_token == 0:
                val.append(0)
            tasks[i].status = Status.FINISHED
            tasks[i].collect(val)
            tasks[i].finish()
        self.tasks = [t for t in self.tasks if t.status != Status.FINISHED]
        log.info(f"Collect finished in {time.time()-t0:.2f}s")