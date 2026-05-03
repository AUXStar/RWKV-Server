from ..reference.sampler.sampler import Sampler


class BatchSampler:
    def __init__(self, sampler=Sampler):
        self.sampler = sampler()

    def setup_rand(self, seed, batch):
        return self.sampler.setup_rand(seed, batch)

    def sample(self, *args, **kwargs):
        """
        执行带**存在惩罚**和**重复惩罚**的批量采样（支持向量化参数）。

        Args:
            logits (torch.Tensor): 输入 Logits 张量，形状要求同 `sample`。
            penalties (torch.Tensor): 惩罚状态张量，形状 (BatchSize, VocabSize)，数据类型 float32，设备 CUDA。
                用于记录历史 Token 的惩罚值，**会被原地更新**。
            states (torch.Tensor): 来自 `setup_rand` 的随机状态张量。
            temperatures (Union[float, list, torch.Tensor], optional): 采样温度，同 `sample`。默认 0.2。
            top_ks (Union[int, list, torch.Tensor], optional): Top-K 参数，同 `sample`。默认 20。
            top_ps (Union[float, list, torch.Tensor], optional): Top-P 参数，同 `sample`。默认 0.7。
            presence_penalties (Union[float, list, torch.Tensor], optional): 存在惩罚系数，范围 [0.0, 10.0]。
                若 Token 曾出现过，对其 Logits 减去固定值（鼓励多样性）。支持 per-sample 设置。默认 0.2。
            repetition_penalties (Union[float, list, torch.Tensor], optional): 重复惩罚系数，范围 [1.0, 10.0]。
                若 Token 曾出现过，对其 Logits 除以该值（抑制重复）。支持 per-sample 设置。默认 1.3。
            penalty_decays (Union[float, list, torch.Tensor], optional): 惩罚衰减因子，范围 [0.0, 1.0]。
                每次采样后，历史惩罚值乘以该因子（逐渐遗忘旧 Token）。支持 per-sample 设置。默认 0.92。
            eos_mask (Union[bool, list, torch.Tensor], optional): EOS（结束符）掩码。
                若为 True，对应样本会强制采样 EOS。支持 per-sample 设置。默认 False。

        Returns:
            torch.Tensor: 采样得到的 Token 索引，形状 (BatchSize,)，数据类型 int32，设备 CUDA。

        Raises:
            ValueError: 当 Logits 维度不支持、数据类型非 float32、V 不符合要求或参数超出范围时。

        Note:
            此函数会**原地更新** `penalties` 张量，用于下一次采样的惩罚计算。

        Performs batch sampling with **presence penalty** and **repetition penalty** (supports vectorized parameters).
        """
        return self.sampler.sample_repetition(*args, **kwargs)
