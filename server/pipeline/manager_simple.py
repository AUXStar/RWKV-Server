# 正常批次推理
import torch
import time
from loguru import logger

from server.pipeline.loader import RWKV070ModelLoader
from server.pipeline.manager_base import TaskManagerBase, WorkerState
from .sampler import BatchSampler
from .task import Status, Task

log = logger.bind(module="taskmanager.single")


class SimpleTaskManager(TaskManagerBase):
    def __init__(
        self,
        model_loader: RWKV070ModelLoader,
        max_batch_size: int = 15,
        buffer_size: int = 100,
    ):
        super().__init__(model_loader, BatchSampler, max_batch_size, buffer_size)
        self.worker = self._init_worker_slots(self.max_batch_size)
        log.info(f"Init capacity = {self.max_batch_size} (fixed)")

    def clear_worker(self):
        self._clear_worker(self.worker)

    def update_batch(self, mask: torch.Tensor):
        free_slots = torch.where(mask)[0].tolist()
        batch_tasks = [task for task in self.tasks if task.status == Status.READY]

        num_assign = min(len(free_slots), len(batch_tasks))
        worker = self.worker

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

    def run(self):
        mask = torch.ones(self.max_batch_size, dtype=torch.bool, device="cuda")
        it = 0
        while self.tasks:
            it += 1
            self.update_batch(mask)
            if mask.all():
                break
            st = time.time()
            self._task_single(mask)
            pulse_time = time.time() - st
            self._collect()

            active = (~mask).sum().item()
            speed = active * self.buffer_size / pulse_time if active else 0
            log.info(
                f"Iter {it} cap={self.max_batch_size} active={active} speed={speed:.2f} tok/s"
            )

    @torch.no_grad()
    def _task_single(self, raw_mask: torch.Tensor):
        t0 = time.time()
        worker: WorkerState = self.worker
        tasks = worker["tasks"]
        last_tokens = worker["last_tokens"]
        state0, state1 = worker["state"]
        penalties, rand_state = worker["penalties"], worker["rand_state"]
        presence_penalties = worker["presence_penalties"]
        repetition_penalties = worker["repetition_penalties"]
        penalty_decays = worker["penalty_decays"]
        temperatures = worker["temperatures"]
        top_ps, top_ks = worker["top_ps"], worker["top_ks"]
        max_tokens = worker["max_tokens"]
        generated_tokens = worker["generated_tokens"]
        max_bs = self.max_batch_size
        mask = raw_mask

        state0 *= ~mask.view(1, 1, max_bs, 1)
        state1 *= ~mask.view(1, max_bs, 1, 1, 1)

        for slot_i in range(self.buffer_size):
            logits = self.model.model.patch_forward_seq_batch(
                last_tokens[: self.max_batch_size].unsqueeze(-1), (state0, state1)
            )
            tokens = self.sampler.sample(
                logits=logits,
                penalties=penalties,
                states=rand_state,
                presence_penalties=presence_penalties,
                repetition_penalties=repetition_penalties,
                penalty_decays=penalty_decays,
                temperatures=temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
                eos_mask=mask,
            )

            generated_tokens[mask, slot_i] = 0
            a = tokens == 0
            b = (last_tokens == 261) & (tokens == 24281)
            mask_tmp = a | b
            tokens[b] = 0

            max_tokens[~mask] -= 1
            mask_new = (mask_tmp | (max_tokens <= 0)) & ~mask
            generated_tokens[~mask, slot_i] = tokens[~mask]

            mask |= mask_new
            state0 *= ~mask_new.view(1, 1, max_bs, 1)
            state1 *= ~mask_new.view(1, max_bs, 1, 1, 1)

            if mask_new.any():
                finished_indices = torch.where(mask_new)[0].tolist()
                for idx in finished_indices:
                    task: Task = tasks[idx]
                    task.current_token = tokens[idx].item()
                    task.state0.copy_(state0[:, :, idx])
                    task.state1.copy_(state1[:, idx])
                    task.penalties.copy_(penalties[idx])
                    task.rand_state.copy_(rand_state[idx * 64 : (idx + 1) * 64])

            tokens[mask] = 0
            last_tokens.copy_(tokens)
            if mask.all():
                generated_tokens[mask, slot_i:] = 0
                break

        raw_mask.copy_(mask)
        log.info(f"Pulse {slot_i+1} steps in {time.time()-t0:.2f}s")

    def _collect(self):
        t0 = time.time()
        generated_tokens = self.worker["generated_tokens"]
        tasks = self.worker["tasks"]
        all_tmp = generated_tokens[:, : self.buffer_size]

        non_zero_mask = all_tmp != 0
        row_any = non_zero_mask.any(dim=1)
        row_all = non_zero_mask.all(dim=1)

        all_nonzero_indices = torch.where(row_all)[0].cpu().numpy()
        mixed_indices = torch.where(row_any & ~row_all)[0].cpu().numpy()

        for i in all_nonzero_indices:
            tasks[i].collect(all_tmp[i].tolist())
        for i in mixed_indices:
            tmp = all_tmp[i]
            non_zero_vals = tmp[tmp != 0].tolist()
            if tasks[i].current_token == 0:
                non_zero_vals.append(0)
            tasks[i].status = Status.FINISHED
            tasks[i].collect(non_zero_vals)
            tasks[i].finish()

        self.tasks = [task for task in self.tasks if task.status != Status.FINISHED]
        log.info(f"Collect finished in {time.time()-t0:.2f}s")
