from ..reference import rwkv7
import torch


class PatchedRWKV7(rwkv7.RWKV_x070):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @rwkv7.MyFunction
    @torch.jit.export
    def patch_forward_seq_batch(
        self, idxs, state: list[torch.Tensor], full_output: bool = False
    ):
        with torch.no_grad():
            z = self.z
            x = z["emb.weight"][idxs]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f"blocks.{i}."
                att = f"blocks.{i}.att."
                ffn = f"blocks.{i}.ffn."

                xx = rwkv7.F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=z[bbb + "ln1.weight"],
                    bias=z[bbb + "ln1.bias"],
                )

                xx, v_first = rwkv7.RWKV_x070_TMix_seq_batch(
                    i,
                    self.n_head,
                    self.head_size,
                    xx,
                    state[0][i],
                    v_first,
                    state[1][i],
                    z[att + "x_r"],
                    z[att + "x_w"],
                    z[att + "x_k"],
                    z[att + "x_v"],
                    z[att + "x_a"],
                    z[att + "x_g"],
                    z[att + "w0"],
                    z[att + "w1"],
                    z[att + "w2"],
                    z[att + "a0"],
                    z[att + "a1"],
                    z[att + "a2"],
                    z[att + "v0"],
                    z[att + "v1"],
                    z[att + "v2"],
                    z[att + "g1"],
                    z[att + "g2"],
                    z[att + "k_k"],
                    z[att + "k_a"],
                    z[att + "r_k"],
                    z[att + "receptance.weight"],
                    z[att + "key.weight"],
                    z[att + "value.weight"],
                    z[att + "output.weight"],
                    z[att + "ln_x.weight"],
                    z[att + "ln_x.bias"],
                )
                x = x + xx

                xx = rwkv7.F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=z[bbb + "ln2.weight"],
                    bias=z[bbb + "ln2.bias"],
                )

                xx = rwkv7.RWKV_x070_CMix_seq_batch(
                    xx,
                    state[0][i],
                    z[ffn + "x_k"],
                    z[ffn + "key.weight"],
                    z[ffn + "value.weight"],
                )
                x = x + xx

            if not full_output:
                x = x[:, -1, :]
            x = rwkv7.F.layer_norm(
                x, (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"]
            )
            x = x @ z["head.weight"]
            return x
