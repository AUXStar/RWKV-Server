import torch
import time
from typing import Optional

# ==================== 假设你已有的CUDA核函数接口 ====================
from server.reference.sampler import Sampler

sampler = Sampler()

def cuda_batch_sample(
    logits: torch.Tensor,          # [B, T, V] float32 CUDA
    penalties: torch.Tensor,       # [B, V] float32 CUDA
    states: torch.Tensor,          # [B] char CUDA（来自setup_rand）
    temperatures: torch.Tensor,    # [B] float32 CUDA
    top_ks: torch.Tensor,          # [B] int32 CUDA
    top_ps: torch.Tensor,          # [B] float32 CUDA
    presence_penalties: torch.Tensor,  # [B] float32 CUDA
    repetition_penalties: torch.Tensor,# [B] float32 CUDA
    penalty_decays: torch.Tensor,      # [B] float32 CUDA
    eos_mask: torch.Tensor,        # [B] bool CUDA
) -> torch.Tensor:                 # [B] int32 CUDA（采样的Token ID）
    return sampler.sample_repetition(
        logits,
        penalties,
        states,
        presence_penalties,
        repetition_penalties,
        penalty_decays,
        temperatures,
        top_ks,
        top_ps,
        eos_mask
    )

def setup_rand(seed, B):
    return sampler.setup_rand(seed, B)

# ==================== 向量化PyTorch CUDA官方采样（对照组） ====================
def torch_gt_sample_vectorized(
    logits: torch.Tensor,          # [B, T, V] float32 CUDA
    temperatures: torch.Tensor,    # [B] float32 CUDA
    top_ks: torch.Tensor,          # [B] int32 CUDA
    top_ps: torch.Tensor,          # [B] float32 CUDA
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """
    向量化PyTorch CUDA官方采样流程（对照组，不含重复惩罚）
    支持任意Batch Size，全CUDA向量化操作，保证数值正确性和结果可复现
    """
    B, T, V = logits.shape
    logits = logits[:, -1, :].clone()  # 取最后一个时间步 [B, V]
    
    # 1. 温度缩放（向量化）
    logits = logits / temperatures[:, None]
    
    # 2. Softmax（数值稳定版，向量化）
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    logits = logits - logits_max
    exp_logits = logits.exp()
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    
    # 3. 批量Top-K截断（向量化）
    # 为了简化，我们取所有样本的最大top_k（实际中可逐样本处理）
    max_top_k = top_ks.max().item()
    if max_top_k > 0 and max_top_k < V:
        top_k_probs, top_k_indices = probs.topk(max_top_k, dim=-1)
    else:
        top_k_probs, top_k_indices = probs, torch.arange(V, device=probs.device).unsqueeze(0).repeat(B, 1)
    
    # 4. 批量Top-P截断（向量化）
    # 先对top_k_probs降序排序
    sorted_probs, sorted_indices = top_k_probs.sort(dim=-1, descending=True)
    # 计算累积概率
    cum_probs = sorted_probs.cumsum(dim=-1)
    # 生成keep_mask：累积概率<=top_p，且至少保留第一个token
    keep_mask = cum_probs <= top_ps[:, None]
    keep_mask[:, 1:] = keep_mask[:, :-1].clone()
    keep_mask[:, 0] = True
    # 用mask截断：将不keep的概率设为0
    keep_probs = sorted_probs * keep_mask.float()
    # 归一化
    keep_probs = keep_probs / keep_probs.sum(dim=-1, keepdim=True)
    
    # 5. 批量采样（向量化）
    # 设置随机种子保证可复现
    generator = torch.Generator(device=probs.device)
    if seed is not None:
        generator.manual_seed(seed)
    # 从keep_probs中采样
    sampled_idx_in_keep = torch.multinomial(keep_probs, 1, generator=generator).squeeze(-1)
    # 映射回原始token索引
    sampled_indices_in_topk = sorted_indices.gather(dim=-1, index=sampled_idx_in_keep.unsqueeze(-1)).squeeze(-1)
    sampled_tokens = top_k_indices.gather(dim=-1, index=sampled_indices_in_topk.unsqueeze(-1)).squeeze(-1)
    
    return sampled_tokens

# ==================== 测速工具函数 ====================
def benchmark_sampler(
    sampler_func,
    inputs: dict,
    warmup_steps: int = 10,
    run_steps: int = 1000,
    device: str = 'cuda',
) -> float:
    """
    测速工具函数：测量采样函数的平均耗时
    :param sampler_func: 采样函数（torch_gt_sample_vectorized或cuda_batch_sample）
    :param inputs: 采样函数的输入参数字典
    :param warmup_steps: 预热步数（避免首次运行的编译/初始化开销）
    :param run_steps: 正式测速步数
    :param device: 设备（'cuda'或'cpu'）
    :return: 平均每次采样的耗时（毫秒ms）
    """
    # 预热
    for _ in range(warmup_steps):
        _ = sampler_func(**inputs)
    
    # 同步设备（确保之前的操作完成）
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 正式测速
    start_time = time.perf_counter()
    for _ in range(run_steps):
        _ = sampler_func(**inputs)
    # 同步设备（确保所有采样操作完成）
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 计算平均耗时
    total_time = end_time - start_time
    avg_time_ms = (total_time / run_steps) * 1000  # 转换为毫秒
    return avg_time_ms

# ==================== 对照试验1：候选ID索引错误（向量化PyTorch CUDA vs 有缺陷CUDA核函数） ====================
def test_candidate_id_error_vectorized():
    print("=== 对照试验1：候选ID索引错误（向量化PyTorch CUDA vs 有缺陷CUDA核函数） ===")
    
    # 1. 构造输入（全CUDA，支持批量）
    B, T, V = 1, 1, 8
    logits = torch.full((B, T, V), -100.0, device='cuda', dtype=torch.float32)
    logits[:, :, 1] = 100.0  # 只有token 1概率≈1
    
    penalties = torch.zeros((B, V), device='cuda', dtype=torch.float32)
    states = setup_rand(seed=42, B=B)
    
    # 2. 采样参数（全CUDA）
    temperatures = torch.tensor([1.0] * B, device='cuda')
    top_ks = torch.tensor([8] * B, device='cuda', dtype=torch.int32)
    top_ps = torch.tensor([1.0] * B, device='cuda')
    presence_penalties = torch.tensor([0.0] * B, device='cuda')
    repetition_penalties = torch.tensor([0.0] * B, device='cuda')
    penalty_decays = torch.tensor([1.0] * B, device='cuda')
    eos_mask = torch.tensor([False] * B, device='cuda', dtype=torch.bool)
    
    # 3. 对照组采样（向量化PyTorch CUDA）
    print("\n--- 对照组（向量化PyTorch CUDA官方实现） ---")
    torch_inputs = {
        'logits': logits,
        'temperatures': temperatures,
        'top_ks': top_ks,
        'top_ps': top_ps,
        'seed': 42,
    }
    # 正确性验证
    torch_outputs = []
    for _ in range(100):
        out = torch_gt_sample_vectorized(**torch_inputs)
        torch_outputs.extend(out.cpu().tolist())
    torch_counts = torch.tensor(torch_outputs).bincount()
    print(f"采样结果统计：{torch_counts}")
    print(f"是否全选1：{all(x == 1 for x in torch_outputs)}")
    # 测速
    torch_avg_time = benchmark_sampler(
        sampler_func=torch_gt_sample_vectorized,
        inputs=torch_inputs,
        warmup_steps=10,
        run_steps=1000,
        device='cuda',
    )
    print(f"平均采样耗时：{torch_avg_time:.4f} ms")
    
    # 4. 试验组采样（有缺陷的CUDA核函数）
    print("\n--- 试验组（有缺陷的CUDA核函数） ---")
    cuda_inputs = {
        'logits': logits,
        'penalties': penalties,
        'states': states,
        'temperatures': temperatures,
        'top_ks': top_ks,
        'top_ps': top_ps,
        'presence_penalties': presence_penalties,
        'repetition_penalties': repetition_penalties,
        'penalty_decays': penalty_decays,
        'eos_mask': eos_mask,
    }
    # 正确性验证
    cuda_outputs = []
    for _ in range(100):
        out = cuda_batch_sample(**cuda_inputs)
        cuda_outputs.extend(out.cpu().tolist())
    cuda_counts = torch.tensor(cuda_outputs).bincount()
    print(f"采样结果统计：{cuda_counts}")
    print(f"是否全选0：{all(x == 0 for x in cuda_outputs)}")
    # 测速
    cuda_avg_time = benchmark_sampler(
        sampler_func=cuda_batch_sample,
        inputs=cuda_inputs,
        warmup_steps=10,
        run_steps=1000,
        device='cuda',
    )
    print(f"平均采样耗时：{cuda_avg_time:.4f} ms")
    
    # 5. 结果对比
    print("\n--- 试验结论 ---")
    if all(x == 1 for x in torch_outputs) and all(x == 0 for x in cuda_outputs):
        print("✅ 验证成功：CUDA核函数候选ID索引错误，永远选0！")
    else:
        print("❌ 验证失败：结果不符合预期。")
    
    # 6. 性能对比
    print("\n--- 性能对比（均为CUDA向量化） ---")
    print(f"向量化PyTorch CUDA平均耗时：{torch_avg_time:.4f} ms")
    print(f"有缺陷CUDA核函数平均耗时：{cuda_avg_time:.4f} ms")
    if cuda_avg_time < torch_avg_time:
        speedup = torch_avg_time / cuda_avg_time
        print(f"⚡ 有缺陷CUDA核函数比向量化PyTorch CUDA快 {speedup:.2f}x（但结果错误！）")
    else:
        slowdown = cuda_avg_time / torch_avg_time
        print(f"⚠️ 有缺陷CUDA核函数比向量化PyTorch CUDA慢 {slowdown:.2f}x（且结果错误！）")

# ==================== 运行所有试验 ====================
if __name__ == "__main__":
    test_candidate_id_error_vectorized()