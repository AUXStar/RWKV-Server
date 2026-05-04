import os
import torch
import time

from .patch import PatchedRWKV7
from ..reference import rwkv7, utils

reference_path = os.path.dirname(os.path.abspath(rwkv7.__file__))


class RWKV070ModelLoader:
    def __init__(
        self,
        model_path: str,
        vocab_path: str = None,
        head_size: int = 64,
        vocab_size: int = 65536,
    ):
        if not vocab_path:
            vocab_path = os.path.join(reference_path, "rwkv_vocab_v20230424.txt")
        load_time = time.time()
        args = type(
            "Args",
            (),
            {
                "MODEL_NAME": model_path,
                "head_size": head_size,
                "vocab_size": vocab_size,
            },
        )()
        self.model = PatchedRWKV7(args)
        self.tokenizer = utils.TRIE_TOKENIZER(vocab_path)
        # 预定义 EOS 检测常量（速度优化）
        self._eos_single_token = torch.tensor(0, dtype=torch.int32, device="cuda")
        self._eos_pair_prev = torch.tensor(261, dtype=torch.int32, device="cuda")
        self._eos_pair_cur = torch.tensor(24281, dtype=torch.int32, device="cuda")
        print(
            f"[Pipeline] Initialized in {time.time() - load_time:.2f} seconds. Model path: {model_path}"
        )

    def gen_state(self, batch_size: int = 0) -> list[torch.Tensor]:
        state = self.model.generate_zero_state(batch_size)
        return state

    def raw_encode(self, text) -> list[int]:
        return self.tokenizer.encode(text)

    def encode(self, text: list[int] | str) -> list[int]:
        if isinstance(text, str):
            return self.raw_encode(text)
        return text

    def raw_decode(self, toks) -> str:
        return self.tokenizer.decode(toks)

    def decode(self, toks: list[int] | str) -> list[int]:
        if isinstance(toks, list):
            return self.raw_decode(toks)
        return toks

    # ---------- 向量化 EOS 检测（GPU 高性能） ----------
    def batch_is_eos(self, prev_tokens: torch.Tensor, cur_tokens: torch.Tensor) -> torch.Tensor:
        """
        批量检测 EOS。
        Args:
            prev_tokens: (B,) int32 on cuda, 上一个 token
            cur_tokens: (B,) int32 on cuda, 当前生成的 token
        Returns:
            (B,) bool tensor on cuda, True 表示该样本应结束
        """
        # 条件1: cur_token == 0
        cond1 = cur_tokens == self._eos_single_token
        # 条件2: prev_token == 261 and cur_token == 24281
        cond2 = (prev_tokens == self._eos_pair_prev) & (cur_tokens == self._eos_pair_cur)
        return cond1 | cond2