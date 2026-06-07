import torch
import time
from loguru import logger

from typing import TypedDict

from ..task import Task

from ..reference import RWKV7

log = logger.bind(module="engine.infer")


class WorkerState(TypedDict):
    tasks: list[Task]
    last_tokens: torch.Tensor
    state0: torch.Tensor
    state1: torch.Tensor
    penalties: torch.Tensor
    rand_state: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    penalty_decays: torch.Tensor
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    max_tokens: torch.Tensor
    generated_tokens: torch.Tensor


class InferEngine:
    __slots__ = (
        "model",
        "sampler",
        "buffer_size",
        "eos_fn",
        "temp_shift_state",
        "temp_wkv_state",
        "temp_elapsed_t",
        "temp_rand_state",
        "temp_penalties",
        "copy_stream",
    )

    def __init__(self, model: RWKV7, sampler, buffer_size: int, eos_fn):
        self.model = model
        self.sampler = sampler
        self.buffer_size = buffer_size
        self.eos_fn = eos_fn  # 批量 EOS 检测函数
        s, w, e = self.model.zero_state(1)
        self.temp_shift_state = s.cpu().pin_memory()
        self.temp_wkv_state = w.cpu().pin_memory()
        self.temp_elapsed_t = e.cpu().pin_memory()
        self.temp_rand_state = torch.zeros(64, dtype=torch.int8).pin_memory()
        self.temp_penalties = torch.zeros(
            self.model.z["emb.weight"].shape[0], dtype=torch.float32
        ).pin_memory()
        self.copy_stream = torch.cuda.Stream()

    @torch.no_grad()
    def generate(
        self,
        worker_state: WorkerState,
        mask: torch.Tensor,
        stop_flags: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        执行批量生成（原地更新 worker_state 和 mask）。
        请注意，推理是根据mask.shape确定范围！

        Args:
            worker_state: 包含所有 GPU 张量的字典（keys: last_tokens, max_tokens, generated_tokens,
                         state, penalties, rand_state, 各种采样参数）
            mask: (B,) bool, True 表示该槽位已完成（不再处理）
            stop_flags: (B,) bool, 可选，True 表示该槽位被外部停止

        Returns:
            generated_tokens 的引用（worker_state["generated_tokens"] 已被原地填充）
        """
        t_start = time.time()
        cur_batch = mask.shape[0]
        w = worker_state

        last = w["last_tokens"][:cur_batch]
        max_tok = w["max_tokens"][:cur_batch]
        gen = w["generated_tokens"][:cur_batch]
        pen = w["penalties"][:cur_batch]
        rand = w["rand_state"][: cur_batch * 64]

        p_pen = w["presence_penalties"][:cur_batch]
        r_pen = w["repetition_penalties"][:cur_batch]
        p_decay = w["penalty_decays"][:cur_batch]
        temp = w["temperatures"][:cur_batch]
        top_p = w["top_ps"][:cur_batch]
        top_k = w["top_ks"][:cur_batch]

        shift_state = w["shift_state"]
        wkv_state = w["wkv_state"]
        elapsed_full = w["elapsed_t"]

        # 初始 mask：已完成槽位不再参与前向
        # 将已完成槽位的 state 清零（避免影响后续可能复用的槽位）
        mask_view = mask.view(1, 1, cur_batch, 1)
        shift_state[:, :, :cur_batch, :] *= ~mask_view
        wkv_state[:, :cur_batch, ...] *= ~mask.view(1, cur_batch, 1, 1, 1)
        elapsed_full[:cur_batch] *= ~mask.view(cur_batch)

        
        if stop_flags is None:
            stop_flags = torch.zeros_like(mask, dtype=torch.bool)

        step = 0
        for step in range(self.buffer_size):
            s0 = shift_state[:, :, :cur_batch, :]
            s1 = wkv_state[:, :cur_batch, ...]
            s3 = elapsed_full[:cur_batch]
            logits = self.model.forward(last.unsqueeze(-1), [s0, s1, s3])
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
            # 已完成的槽位不记录生成 token
            gen[mask, step] = 0
            # EOS 检测
            finish = self.eos_fn(last, tokens)
            # 特殊处理 eos2 时把 token 置为 0（保持习惯）
            tokens[finish & (tokens == 24281)] = 0

            # 更新剩余 token 计数
            max_tok[~mask] -= 1
            # 新增停止标志
            stop_cond = stop_flags[:cur_batch] & ~mask

            new_finish = (finish | (max_tok <= 0) | stop_cond) & ~mask

            # 记录生成的 token，仅未完成的槽位
            gen[~mask, step] = tokens[~mask]

            # 更新 state，已完成槽位的 state 清零
            if new_finish.any():
                # 将完成的任务状态回写到 Task
                finished_indices = torch.where(new_finish)[0].tolist()
                tasks = w["tasks"]
                for idx in finished_indices:
                    task: Task = tasks[idx]
                    if task is None:
                        continue
                    task.current_token = tokens[idx].item()
                    self.temp_shift_state.copy_(shift_state[:, :, [idx]])
                    self.temp_wkv_state.copy_(wkv_state[:, [idx]])
                    self.temp_elapsed_t.copy_(elapsed_full[[idx]])
                    self.temp_penalties.copy_(pen[idx])
                    self.temp_rand_state.copy_(rand[idx * 64 : (idx + 1) * 64])

                    with torch.cuda.stream(self.copy_stream):
                        task.shift_state.copy_(self.temp_shift_state)
                        task.wkv_state.copy_(self.temp_wkv_state)
                        task.elapsed_t.copy_(self.temp_elapsed_t)
                        task.penalties.copy_(self.temp_penalties)
                        task.rand_state.copy_(self.temp_rand_state)

                shift_state[:, :, :cur_batch, :] *= ~new_finish.view(1, 1, cur_batch, 1)
                wkv_state[:, :cur_batch, ...] *= ~new_finish.view(1, cur_batch, 1, 1, 1)
                elapsed_full[:cur_batch] *= ~new_finish.view(cur_batch)

            mask |= new_finish
            tokens[mask] = 0
            last.copy_(tokens)

            if mask.all():
                gen[mask, step + 1 :] = 0
                break

        elapsed = time.time() - t_start
        log.info(f"Pulse {step+1} steps in {elapsed:.2f}s")
        return gen
