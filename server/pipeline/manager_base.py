from abc import ABC, abstractmethod
from typing import List, Optional
import torch

from .loader import RWKV070ModelLoader
from .sampler import BatchSampler
from .task import Task, Status


class BaseScheduler(ABC):
    def __init__(
        self,
        model_loader: RWKV070ModelLoader,
        batch_sampler: BatchSampler,
        max_batch_size: int,
        buffer_size: int = 100,
    ):
        self.model = model_loader
        self.sampler = batch_sampler()
        self.max_batch_size = max_batch_size
        self.buffer_size = buffer_size
        self.tasks: List[Task] = []
        self.finished_tasks: List[Task] = []

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
            model_loader=self.model,
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
        self.tasks.append(task)
        return task

    def _init_worker_slots(self, batch_size: int) -> dict:
        vocab_size = self.model.model.args.vocab_size
        state0, state1 = self.model.gen_state(batch_size)
        return {
            "last_tokens": torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
            "max_tokens": torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
            "tasks": [None] * batch_size,
            "generated_tokens": torch.zeros(
                (batch_size, self.buffer_size), dtype=torch.int32, device="cuda"
            ),
            "state0": state0,
            "state1": state1,
            "penalties": torch.zeros(batch_size, vocab_size, dtype=torch.float32, device="cuda"),
            "rand_state": torch.zeros(64 * batch_size, dtype=torch.int8, device="cuda"),
            "presence_penalties": torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "repetition_penalties": torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "penalty_decays": torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "temperatures": torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "top_ps": torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "top_ks": torch.zeros(batch_size, dtype=torch.int, device="cuda"),
        }

    def _clear_worker(self, worker: dict):
        worker["tasks"] = [None] * len(worker["tasks"])

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def update_batch(self, mask: torch.Tensor):
        pass

    @abstractmethod
    def _collect(self):
        pass