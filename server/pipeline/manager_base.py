# task_manager_base.py
from abc import ABC, abstractmethod
from typing import List, Optional
import torch

from .loader import RWKV070ModelLoader
from .sampler import BatchSampler
from .task import Task, Status

from typing import TypedDict, Tuple


class WorkerState(TypedDict):
    tasks: list[Task]
    last_tokens: torch.Tensor
    state0: torch.Tensor
    state1: torch.Tensor
    penalties: torch.Tensor
    rand_state: torch.Tensor  # 或者是 torch.Generator，看你具体类型
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    penalty_decays: torch.Tensor
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    max_tokens: torch.Tensor
    generated_tokens: torch.Tensor


class TaskManagerBase(ABC):
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

    # ========== 公共任务管理 ==========
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
        """创建一个新任务，加入队列，返回Task对象"""
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

    # ========== 槽位初始化 (供子类调用) ==========
    def _init_worker_slots(self, batch_size: int) -> dict:
        """
        创建一个 worker 所需的所有 GPU 张量。
        返回字典，子类可将其存入 self.worker 或 self.workers 列表中。
        """
        vocab_size = self.model.model.args.vocab_size
        state0, state1 = self.model.gen_state(
            batch_size
        )  # state0, state1 已经是 list[Tensor]
        # 注意：gen_state 返回的 state0 形状可能是 (1, 1, batch, head_size) 等，由模型决定
        return {
            "last_tokens": torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
            "max_tokens": torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
            "tasks": [None] * batch_size,
            "generated_tokens": torch.zeros(
                (batch_size, self.buffer_size), dtype=torch.int32, device="cuda"
            ),
            "state": [state0, state1],
            "penalties": torch.zeros(
                batch_size, vocab_size, dtype=torch.float32, device="cuda"
            ),
            "rand_state": torch.zeros(64 * batch_size, dtype=torch.int8, device="cuda"),
            # 采样参数
            "presence_penalties": torch.zeros(
                batch_size, dtype=torch.float16, device="cuda"
            ),
            "repetition_penalties": torch.zeros(
                batch_size, dtype=torch.float16, device="cuda"
            ),
            "penalty_decays": torch.zeros(
                batch_size, dtype=torch.float16, device="cuda"
            ),
            "temperatures": torch.zeros(batch_size, dtype=torch.float16, device="cuda"),
            "top_ps": torch.zeros(batch_size, dtype=torch.float16, device="cuda"),
            "top_ks": torch.zeros(batch_size, dtype=torch.int, device="cuda"),
        }

    def _clear_worker(self, worker: dict):
        """将 worker 字典中的所有任务槽位重置（可选）"""
        # 实际使用中，只需将 tasks 列表置 None，其他张量会被后续覆盖
        worker["tasks"] = [None] * len(worker["tasks"])
        # 其他张量可以保持原值，但逻辑上需要重置 last_tokens 等？不需要，因为它们会在分配新任务时被覆盖
        # 但为了安全，可以重置 last_tokens 为 0
        # worker["last_tokens"].zero_()
        # worker["max_tokens"].zero_()
        # worker["generated_tokens"].zero_()
        # 注意：state 和 penalties 等不会被清零，因为会被新任务覆盖

    # ========== 抽象方法（子类必须实现） ==========
    @abstractmethod
    def run(self):
        """主循环：持续处理任务直到队列为空"""
        pass

    @abstractmethod
    def update_batch(self, mask: torch.Tensor) -> torch.Tensor:
        """
        根据空闲槽位 mask，将 READY 状态的任务分配到对应的 worker 槽位中。
        返回更新后的 mask。
        """
        pass

    @abstractmethod
    def _task_single(self, mask: torch.Tensor):
        """
        对一个 worker 执行一次“脉冲”：生成 buffer_size 个 token 或直到所有任务完成。
        通常会在子类中被多线程调用。
        """
        pass

    @abstractmethod
    def _collect(self):
        """
        从 worker 的 generated_tokens 中收集结果，调用任务的 callback，
        并将已完成的任务移出队列。
        """
        pass
