import time
import numpy as np
from typing import List
import torch

# ======================
# 1. 统一测试配置
# ======================
TEST_CONFIG = {
    "prompt_count": 190,        # 测试用的prompt数量（可调整）
    "max_batch_size": 35,      # 和你代码保持一致
    "buffer_size": 100,
    "model_path": "/home/njzy/rwkv_agent/server/model/rwkv7-g1f-2.9b-20260420-ctx8192",
    "infer_params": {
        "max_tokens": 2000,
        "presence_penalty": 2,
        "repetition_penalty": 0,
        "penalty_decay": 1.0,
        "temperature": 1,
        "top_k": -1,
        "top_p": 0.5,
    }
}

# 生成测试用 prompts（模拟真实输入）
def generate_test_prompts(count: int = 50) -> List[str]:
    return [f"请分析以下内容：测试文本{i}，请用中文回答" for i in range(count)]

# ======================
# 2. 统一性能测试函数
# ======================
def run_pipeline_benchmark(
    pipeline_name: str,
    pipeline_module,
    prompts: List[str],
    config: dict
):
    print(f"\n{'='*50}")
    print(f"🚀 开始测试 {pipeline_name}")
    print(f"{'='*50}")
    
    # 1. 初始化 pipeline + task_manager
    t_init_start = time.time()
    pipeline = pipeline_module.RWKV_070_Pipeline(config["model_path"])
    task_manager = pipeline_module.TaskManager(
        pipeline, 
        max_batch_size=config["max_batch_size"], 
        buffer_size=config["buffer_size"]
    )
    t_init_end = time.time()
    init_time = t_init_end - t_init_start
    print(f"✅ {pipeline_name} 初始化完成，耗时：{init_time:.2f}s")

    # 2. 提交所有任务到队列
    t_submit_start = time.time()
    ttts = [
        task_manager.new_task(
            f"User: {p}\n\nAssistent: <thinking>好的，用户的语言是",
            **config["infer_params"]
        )
        for p in prompts
    ]
    t_submit_end = time.time()
    submit_time = t_submit_end - t_submit_start
    print(f"✅ 任务提交完成，总数：{len(prompts)}，提交耗时：{submit_time:.4f}s")

    # 3. 等待第一个任务进入就绪（模拟你原代码逻辑）
    t_first_wait_start = time.time()
    while ttts[0].status != pipeline_module.Status.READY:
        time.sleep(0.001)  # 避免空转占CPU
    t_first_wait_end = time.time()
    first_task_wait_time = t_first_wait_end - t_first_wait_start
    print(f"✅ 首个任务就绪，等待耗时：{first_task_wait_time:.4f}s")
    time.sleep(5)
    # 4. 执行推理（核心性能阶段）
    t_run_start = time.time()
    task_manager.run()
    t_run_end = time.time()
    total_infer_time = t_run_end - t_run_start
    avg_task_time = total_infer_time / len(prompts)
    
    # 5. 统计总生成 token（RWKV 任务通常有 .total_tokens 属性）
    total_tokens = 0
    for task in ttts:
        try:
            total_tokens += task.total_tokens  # 根据你的框架适配
        except:
            total_tokens += config["infer_params"]["max_tokens"]  # 兜底
    
    throughput = total_tokens / total_infer_time if total_infer_time > 0 else 0

    # ======================
    # 输出性能报告
    # ======================
    print(f"\n📊 {pipeline_name} 性能测试报告")
    print(f"-" * 40)
    print(f"初始化耗时        : {init_time:.2f} s")
    print(f"任务提交耗时        : {submit_time:.4f} s")
    print(f"首任务等待耗时      : {first_task_wait_time:.4f} s")
    print(f"总推理耗时          : {total_infer_time:.2f} s")
    print(f"单任务平均耗时      : {avg_task_time:.3f} s")
    print(f"总生成 Token 数     : {total_tokens}")
    print(f"推理吞吐量          : {throughput:.2f} token/s")
    print(f"批量大小            : {config['max_batch_size']}")
    print(f"任务总数            : {len(prompts)}")
    
    return {
        "name": pipeline_name,
        "init_time": init_time,
        "submit_time": submit_time,
        "first_task_wait": first_task_wait_time,
        "total_infer_time": total_infer_time,
        "avg_task_time": avg_task_time,
        "total_tokens": total_tokens,
        "throughput": throughput,
    }

# ======================
# 3. 执行对比测试
# ======================
if __name__ == "__main__":
    # 导入你的两个 pipeline
    from server.pipeline import pipeline as pipelineA
    from server.pipeline import pipeline1 as pipelineB

    # 生成测试数据
    prompts = generate_test_prompts(TEST_CONFIG["prompt_count"])

    # 测试 A
    result_a = run_pipeline_benchmark("PipelineA", pipelineA, prompts, TEST_CONFIG)
    torch.cuda.empty_cache()

    # 测试 B
    result_b = run_pipeline_benchmark("PipelineB", pipelineB, prompts, TEST_CONFIG)

    # ======================
    # 最终对比总报告
    # ======================
    print("\n\n" + "="*60)
    print("🏆 PipelineA vs PipelineB 最终性能对比")
    print("="*60)
    print(f"{'指标':<18}{'PipelineA':<15}{'PipelineB':<15}{'差异'}")
    print("-"*60)
    
    metrics = [
        ("初始化耗时(s)", "init_time", ""),
        ("总推理耗时(s)", "total_infer_time", "↓更快"),
        ("单任务平均(s)", "avg_task_time", "↓更快"),
        ("吞吐量(token/s)", "throughput", "↑更高"),
        ("首任务等待(s)", "first_task_wait", "↓更快"),
    ]
    
    for label, key, desc in metrics:
        va = result_a[key]
        vb = result_b[key]
        diff = vb - va
        diff_pct = (diff / va) * 100 if va !=0 else 0
        mark = "🔼" if diff>0 and "耗时" in label else ("🔽" if diff<0 and "耗时" in label else "")
        print(f"{label:<18}{va:<15.2f}{vb:<15.2f}{diff:+.2f}s({diff_pct:+.1f}%) {mark} {desc}")