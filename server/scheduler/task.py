import threading
from typing import Iterable

import torch
import uuid
import enum

from ..state_manager import state_pool
from .loader import RWKV070ModelLoader
from .sampler import BatchSampler


class Status(enum.Enum):
    PREFILL = 0
    READY = 1
    RUNNING = 2
    FINISHED = 3
    STOP = 4


class NullLock:
    """空锁，用于不需要同步的场景"""

    def acquire(self):
        pass

    def release(self):
        pass


class Task:
    state0: torch.Tensor
    state1: torch.Tensor
    rand_state: torch.Tensor
    penalties: torch.Tensor
    current_token: int

    def __init__(
        self,
        prompt: str | list[int],
        model_loader: RWKV070ModelLoader,
        batch_sampler: BatchSampler,
        max_tokens: int = 2048,
        presence_penalty: float = 2,
        repetition_penalty: float = 0,
        penalty_decay: float = 0.994,
        temperature: float = 1,
        top_k: int = 20,
        top_p: float = 0.5,
        seed: int = 42,
        collect_callback=None,
        finish_callback=None,
        lock=None,
        async_prefill: bool = False,
    ):
        self.collect_callback = collect_callback or self._default_collect_callback
        self.finish_callback = finish_callback or self._default_finish_callback
        self.shift_state, self.wkv_state, self.elapsed_t = model_loader.gen_state()
        self.rand_state = batch_sampler.setup_rand(seed, 1)
        self.penalties = torch.zeros(model_loader.vocab_size, dtype=torch.float32)
        self.model_loader = model_loader
        self.lock = lock if lock is not None else NullLock()

        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.penalty_decay = penalty_decay
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.generated_tokens = []
        self.current_token = -1
        self.status = Status.PREFILL
        self.stop_flag_tensor = None  # 将被 manager 绑定到 GPU 张量

        self.prefill_thread = None
        if async_prefill:
            self._start_async_prefill(prompt)
        else:
            self._sync_prefill(prompt)
            self.cpu()

    # ---------- 预填充逻辑（原样保留） ----------
    def _sync_prefill(self, prompt: Iterable[int] | str):
        prompt = self.tokenize(prompt)
        self.generated_tokens += prompt
        assert len(prompt) >= 1 and all(isinstance(i, int) for i in prompt)
        if self.current_token != -1:
            prompt.insert(0, self.current_token)
        if len(prompt) >= 2:
            if self.shift_state.device != "cuda":
                self.cuda()
            tokens = torch.tensor(prompt[:-1], dtype=torch.long, device="cpu")
            self.lock.acquire()
            self.model_loader.model.forward(tokens,(self.shift_state, self.wkv_state, self.elapsed_t))
            self.lock.release()
        self.current_token = prompt[-1]
        self.status = Status.READY

    def _start_async_prefill(self, prompt: Iterable[int] | str):
        def _prefill_worker():
            prompt_tokens = self.tokenize(prompt)
            self.generated_tokens += prompt_tokens
            assert len(prompt_tokens) >= 1 and all(
                isinstance(i, int) for i in prompt_tokens
            )
            if self.current_token != -1:
                prompt_tokens.insert(0, self.current_token)
            if len(prompt_tokens) >= 2:
                need_cuda = self.shift_state.device != "cuda"
                if need_cuda:
                    self.cuda()
                tokens = torch.tensor(prompt_tokens[:-1], dtype=torch.long, device="cpu")
                self.lock.acquire()
                self.model_loader.model.forward(tokens,(self.shift_state, self.wkv_state, self.elapsed_t))
                self.lock.release()
                if need_cuda:
                    self.cpu()
            self.current_token = prompt_tokens[-1]
            self.status = Status.READY

        self.prefill_thread = threading.Thread(target=_prefill_worker)
        self.prefill_thread.start()

    # ---------- 状态保存/加载 ----------
    def save_state(self):
        session_id = uuid.uuid4().hex
        current_token = torch.tensor(self.current_token, dtype=torch.int32)
        state_pool.get_state_manager().put_state(
            session_id,
            [self.shift_state, self.wkv_state, self.elapsed_t, current_token],
        )
        return session_id

    def load_state(self, session_id: str):
        self.shift_state, self.wkv_state, self.elapsed_t, current_token = (
            state_pool.get_state_manager().get_state(session_id)
        )
        self.current_token = current_token.item()
        return session_id

    # ---------- 辅助方法 ----------
    def tokenize(self, prompt: str) -> list[int]:
        if isinstance(prompt, str):
            prompt = self.model_loader.tokenizer.encode(prompt)
        return prompt

    def continue_gen(self):
        self.status = Status.READY
        self.cpu()

    def prefill(self, prompt: Iterable[int] | str):
        self._sync_prefill(prompt)
        self.cpu()

    def prepare(self):
        """被 manager 调用，确保张量在 GPU 上，等待异步预填充完成"""
        if self.prefill_thread and self.prefill_thread.is_alive():
            self.prefill_thread.join()
        self.cuda()

    def stop(self):
        """
        停止任务生成。
        如果已绑定 stop_flag_tensor（由 manager 设置），则直接修改 GPU 张量；
        否则回退到修改 status。
        """
        if self.stop_flag_tensor is not None:
            self.stop_flag_tensor[...] = True
        else:
            self.status = Status.STOP

    def cuda(self):
        self.shift_state = self.shift_state.cuda()
        self.wkv_state = self.wkv_state.cuda()
        self.elapsed_t = self.elapsed_t.cuda()
        self.rand_state = self.rand_state.cuda()
        self.penalties = self.penalties.cuda()
        return self

    def cpu(self):
        self.shift_state = self.shift_state.cpu()
        self.wkv_state = self.wkv_state.cpu()
        self.elapsed_t = self.elapsed_t.cpu()
        self.rand_state = self.rand_state.cpu()
        self.penalties = self.penalties.cpu()
        return self

    def collect(self, tokens: list[int]):
        self.generated_tokens.extend(tokens)
        self.collect_callback(tokens)

    def _default_collect_callback(self, tokens: list[int]):
        try:
            print("#" * 30, "\n", self.model_loader.raw_decode(tokens))
        except Exception:
            print("broken tokens")

    def finish(self):
        self.cpu()
        print(self.generated_tokens)
        self.finish_callback(self.model_loader.raw_decode(self.generated_tokens))

    def _default_finish_callback(self,_):
        print("!" * 30, "\n", self.model_loader.raw_decode(self.generated_tokens))
