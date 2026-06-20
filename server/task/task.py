from typing import Iterable
import torch
import threading
import enum
import time
import loguru
from ..scheduler.loader import RWKV070ModelLoader
from ..scheduler.batch_sampler import BatchSampler
from ..utils import NullLock, nop

log = loguru.logger.bind(module="task")


class Status(enum.Enum):
    PREFILL = 0
    READY = 1
    RUNNING = 2
    FINISHED = 3


class Task:
    run_time: float = 0
    finish_time: float = 0
    prefill_time: float = 0
    per_speed: float = 0

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
        task_id: str = "TMP",
        collect_callback=None,
        finish_callback=None,
        prefill_lock=None,
    ):
        self.task_id = task_id
        self.collect_callback = collect_callback or nop
        self.finish_callback = finish_callback or nop
        self.shift_state, self.wkv_state, self.elapsed_t = model_loader.gen_state()
        self.rand_state = batch_sampler.setup_rand(seed, 1)
        self.penalties = torch.zeros(model_loader.vocab_size, dtype=torch.float32)
        self.model_loader = model_loader
        self.batch_sampler = batch_sampler
        self.prefill_lock = prefill_lock if prefill_lock is not None else NullLock()
        self.tokens_lock = threading.Lock()

        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.penalty_decay = penalty_decay
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed

        self._token_batches: list[list[int]] = []
        self.current_token = -1
        self.stop_flag_tensor = None
        self._status = Status.PREFILL

        self.prefill(prompt)
        self.cpu()

    def info(self):
        return dict(
            task_id = self.task_id,
            generated_buf = sum(len(b) for b in self._token_batches),
            status = self.status,
        )

    def __getstate__(self):
        state = {
            "task_id": self.task_id,
            "shift_state": self.shift_state.cpu(),
            "wkv_state": self.wkv_state.cpu(),
            "elapsed_t": self.elapsed_t.cpu(),
            "rand_state": self.rand_state.cpu(),
            "penalties": self.penalties.cpu(),
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "penalty_decay": self.penalty_decay,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
            "current_token": self.current_token,
            "_status": self._status,
            "_token_batches": self._token_batches,
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model_loader = None
        self.batch_sampler = None
        self.prefill_lock = NullLock()
        self.collect_callback = nop
        self.finish_callback = nop
        self.tokens_lock = threading.Lock()
        self.stop_flag_tensor = None

    def __enter__(self):
        self.prepare()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cpu()

    def prefill(self, prompt: Iterable[int] | str):
        assert self.status != Status.RUNNING

        self._status = Status.PREFILL
        # 每次 prefill 清除旧 token 历史
        with self.tokens_lock:
            self._token_batches.clear()
        prompt = self.tokenize(prompt)

        assert len(prompt) >= 1 and all(isinstance(i, int) for i in prompt)
        t = time.time()
        if self.current_token != -1:
            prompt.insert(0, self.current_token)
        if len(prompt) >= 2:
            if self.shift_state.device != "cuda":
                self.cuda()
            tokens = torch.tensor(prompt[:-1], dtype=torch.long, device="cpu")
            self.prefill_lock.acquire()
            self.model_loader.model.forward(
                tokens, (self.shift_state, self.wkv_state, self.elapsed_t)
            )
            self.prefill_lock.release()
        self.current_token = prompt[-1]
        if self.max_tokens == 0:
            self._status = Status.FINISHED
            self.collect([])
            self.finish()
        else:
            self._status = Status.READY
        self.prefill_time = time.time() - t

    def tokenize(self, prompt: str) -> list[int]:
        if isinstance(prompt, str):
            prompt = self.model_loader.tokenizer.encode(prompt)
        return prompt

    def continue_gen(self):
        self._status = Status.READY
        self.cpu()

    def prepare(self):
        self.cuda()
        self.run_time = time.time()
        self.finish_time = 0

    def stop(self):
        if self.stop_flag_tensor is not None:
            self.stop_flag_tensor[...] = True
        else:
            self._status = Status.FINISHED

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
        with self.tokens_lock:
            self._token_batches.append(tokens)
        self.collect_callback(tokens)

    def get_all_tokens(self) -> list[int]:
        """获取当前所有已生成的 token（不清空）。"""
        # 读取不需要锁（CPython GIL 保证 list.append 原子性）
        all_tokens: list[int] = []
        for batch in self._token_batches:
            all_tokens.extend(batch)
        return all_tokens

    def pop_tokens(self):
        """兼容旧接口：获取并清空。"""
        with self.tokens_lock:
            all_tokens: list[int] = []
            for batch in self._token_batches:
                all_tokens.extend(batch)
            self._token_batches.clear()
        return all_tokens

    def finish(self):
        self.cpu()
        self.finish_time = time.time()
        # 传合并后的完整 token 列表给 finish_callback
        all_tokens: list[int] = []
        for batch in self._token_batches:
            all_tokens.extend(batch)
        self.finish_callback(all_tokens)

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, value: Status):
        self._status = value

    def decode(self, tokens: list[int]) -> str:
        if self.model_loader is None:
            return ""
        return self.model_loader.decode(tokens)
