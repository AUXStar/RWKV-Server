import torch
import time

from ..reference import RWKV7
from rwkv.utils import PIPELINE # from pip module rwkv

class RWKV070ModelLoader:
    def __init__(
        self,
        model_path: str,
        vocab_path: str|None = None,
    ):
        load_time = time.time()
        if not vocab_path:
            vocab_path = "rwkv_vocab_v20230424"
        self.model = RWKV7(model_path)
        self.vocab_size = self.model.z["emb.weight"].shape[0]
        pipeline = PIPELINE(self.model,vocab_path)
        self.tokenizer = pipeline.tokenizer
        self._eos_single_token = torch.tensor(0, dtype=torch.int32, device="cuda")#0
        self._eos_pair_prev0 = torch.tensor(261, dtype=torch.int32, device="cuda") #\n\n
        self._eos_pair_prev1 = torch.tensor(11, dtype=torch.int32, device="cuda") #\n
        self._eos_pair_cur = torch.tensor(24281, dtype=torch.int32, device="cuda")#User
        print(
            f"[Pipeline] Initialized in {time.time() - load_time:.2f} seconds. Model path: {model_path}"
        )

    def gen_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = max(batch_size, 1)
        shift_state, wkv_state, elapsed_t = self.model.zero_state(batch_size)
        return shift_state, wkv_state, elapsed_t

    def raw_encode(self, text) -> list[int]:
        return self.tokenizer.encode(text)

    def encode(self, text: list[int] | str) -> list[int]:
        if isinstance(text, str):
            return self.raw_encode(text)
        return text

    def raw_decode(self, toks) -> str:
        if not toks: return ""
        if not toks[-1]: toks = toks[:-1]
        return self.tokenizer.decode(toks)

    def decode(self, toks: list[int] | str) -> str:
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
        cond2 = (prev_tokens == self._eos_pair_prev0) & (cur_tokens == self._eos_pair_cur)
        # 条件3: prev_token == 11 and cur_token == 24281
        cond3 = (prev_tokens == self._eos_pair_prev1) & (cur_tokens == self._eos_pair_cur)
        return cond1 | cond2 | cond3