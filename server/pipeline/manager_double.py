# 双缓冲，由于模型开销大，该管理器表现较差
import threading
import time
import torch
from loguru import logger

from .loader import RWKV070ModelLoader
from .manager_base import TaskManagerBase,WorkerState
from .sampler import BatchSampler
from .task import Status, Task

log = logger.bind(module="taskmanager.double")


class DoubleBufferTaskManager(TaskManagerBase):
    def __init__(
        self,
        model_loader: RWKV070ModelLoader,
        max_batch_size: int = 15,
        buffer_size: int = 100,
    ):
        super().__init__(model_loader, BatchSampler, max_batch_size, buffer_size)
        self.workers = [
            self._init_worker_slots(self.max_batch_size),
            self._init_worker_slots(self.max_batch_size),
        ]
        self.masks = [
            torch.ones(self.max_batch_size, dtype=torch.bool, device="cuda"),
            torch.ones(self.max_batch_size, dtype=torch.bool, device="cuda"),
        ]
        log.info(f"Init max_batch_size = {max_batch_size} (fixed)")

    def clear_worker(self, worker_id: int):
        self._clear_worker(self.workers[worker_id])

    def update_batch(self, worker_id: int, mask: torch.Tensor) -> torch.Tensor:
        worker = self.workers[worker_id]
        free_slots = torch.where(mask)[0].tolist()
        ready_tasks = [t for t in self.tasks if t.status == Status.READY]
        half = len(ready_tasks) // 2
        if worker_id == 0:
            batch_tasks = ready_tasks[:half]
        else:
            batch_tasks = ready_tasks[half:]

        num_assign = min(len(free_slots), len(batch_tasks))
        for i in range(num_assign):
            slot_id = free_slots[i]
            task = batch_tasks[i]
            task.status = Status.RUNNING
            task.prepare()

            worker["last_tokens"][slot_id] = task.current_token
            worker["max_tokens"][slot_id] = task.max_tokens
            worker["tasks"][slot_id] = task
            worker["state"][0][:, :, slot_id, :] = task.state0
            worker["state"][1][:, slot_id, :, :, :] = task.state1
            worker["penalties"][slot_id] = task.penalties
            worker["rand_state"][slot_id * 64 : (slot_id + 1) * 64] = task.rand_state
            worker["presence_penalties"][slot_id] = task.presence_penalty
            worker["repetition_penalties"][slot_id] = task.repetition_penalty
            worker["penalty_decays"][slot_id] = task.penalty_decay
            worker["temperatures"][slot_id] = task.temperature
            worker["top_ps"][slot_id] = task.top_p
            worker["top_ks"][slot_id] = task.top_k
            mask[slot_id] = False
        return mask

    def run(self):
        it = 0
        while self.tasks:
            it += 1
            self.masks[0] = self.update_batch(0, self.masks[0])
            self.masks[1] = self.update_batch(1, self.masks[1])

            if self.masks[0].all() and self.masks[1].all():
                break

            st = time.time()
            t0 = threading.Thread(target=self._task_single, args=(0, self.masks[0]))
            t1 = threading.Thread(target=self._task_single, args=(1, self.masks[1]))
            t0.start()
            t1.start()
            t0.join()
            t1.join()

            self._collect(0)
            self._collect(1)

            self._clear_worker(self.workers[0])
            self._clear_worker(self.workers[1])

            total_slots = 2 * self.max_batch_size
            active_slots = (self.masks[0] == False).sum() + (
                self.masks[1] == False
            ).sum()
            elapsed = time.time() - st
            speed = active_slots * self.buffer_size / elapsed if elapsed > 0 else 0
            log.info(f"Iter {it} active={active_slots} speed={speed:.2f} tok/s")

    @torch.no_grad()
    def _task_single(self, worker_id: int, raw_mask: torch.Tensor):
        t0 = time.time()
        worker: WorkerState = self.workers[worker_id]
        tasks = worker["tasks"]
        last_tokens = worker["last_tokens"]
        state0, state1 = worker["state"]
        penalties, rand_state = worker["penalties"], worker["rand_state"]
        presence = worker["presence_penalties"]
        repetition = worker["repetition_penalties"]
        penalty_decay = worker["penalty_decays"]
        temperatures = worker["temperatures"]
        top_ps = worker["top_ps"]
        top_ks = worker["top_ks"]
        max_tokens = worker["max_tokens"]
        generated = worker["generated_tokens"]
        max_bs = self.max_batch_size

        mask = raw_mask.clone()
        state0 *= ~mask.view(1, 1, max_bs, 1)
        state1 *= ~mask.view(1, max_bs, 1, 1, 1)

        for step in range(self.buffer_size):
            logits = self.model.model.patch_forward_seq_batch(
                last_tokens[:max_bs].unsqueeze(-1), (state0, state1)
            )
            tokens = self.sampler.sample(
                logits=logits,
                penalties=penalties,
                states=rand_state,
                presence_penalties=presence,
                repetition_penalties=repetition,
                penalty_decays=penalty_decay,
                temperatures=temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
                eos_mask=mask,
            )
            generated[mask, step] = 0
            eos1 = tokens == 0
            eos2 = (last_tokens == 261) & (tokens == 24281)
            finish = eos1 | eos2
            tokens[eos2] = 0

            max_tokens[~mask] -= 1
            new_finish = (finish | (max_tokens <= 0)) & ~mask
            generated[~mask, step] = tokens[~mask]

            if new_finish.any():
                state0 *= ~new_finish.view(1, 1, max_bs, 1)
                state1 *= ~new_finish.view(1, max_bs, 1, 1, 1)
                finished_ids = torch.where(new_finish)[0].tolist()
                for idx in finished_ids:
                    task = tasks[idx]
                    task.current_token = tokens[idx].item()
                    task.state0.copy_(state0[:, :, idx])
                    task.state1.copy_(state1[:, idx])
                    task.penalties.copy_(penalties[idx])
                    task.rand_state.copy_(rand_state[idx * 64 : (idx + 1) * 64])

            mask |= new_finish
            tokens[mask] = 0
            last_tokens.copy_(tokens)

            if mask.all():
                generated[mask, step + 1 :] = 0
                break

        raw_mask.copy_(mask)
        log.info(f"Pulse worker={worker_id} steps={step+1} time={time.time()-t0:.2f}s")

    def _collect(self, worker_id: int):
        t0 = time.time()
        worker = self.workers[worker_id]
        generated = worker["generated_tokens"]
        tasks = worker["tasks"]
        all_tmp = generated[:, : self.buffer_size]

        non_zero = all_tmp != 0
        row_any = non_zero.any(dim=1)
        row_all = non_zero.all(dim=1)

        for i in torch.where(row_all)[0].cpu().numpy():
            tasks[i].collect(all_tmp[i].tolist())
            tasks[i].status = Status.FINISHED
            tasks[i].finish()
        for i in torch.where(row_any & ~row_all)[0].cpu().numpy():
            non_zero_vals = all_tmp[i][all_tmp[i] != 0].tolist()
            if tasks[i].current_token == 0:
                non_zero_vals.append(0)
            tasks[i].collect(non_zero_vals)
            tasks[i].status = Status.FINISHED
            tasks[i].finish()

        self.tasks = [t for t in self.tasks if t.status != Status.FINISHED]
        log.info(f"Collect worker={worker_id} finished in {time.time()-t0:.2f}s")
