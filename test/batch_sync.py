#!/usr/bin/env python3
"""
同步版批量测试 - 使用 requests + ThreadPoolExecutor
能处理含特殊字符的响应，并输出详细错误信息
"""

import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

BASE_URL = "http://localhost:8000"  # 请修改为您的实际地址

def create_task(prompt: str, max_tokens: int = 100) -> str:
    """创建任务，返回 task_id"""
    resp = requests.post(
        f"{BASE_URL}/v1/tasks/create",
        json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    # 兼容多种字段名
    return data.get("task_id") or data.get("id")

def wait_for_result(task_id: str, timeout: float = 180, poll_interval: float = 1.0) -> Dict:
    """轮询直到任务完成，返回结果字典"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/v1/tasks/{task_id}/get_result", timeout=10)
            if resp.status_code == 200:
                # 直接使用 .json() 解析，它能处理一些控制字符
                data = resp.json()
                if data.get("finished") is True:
                    return data
                # 如果服务端返回 finished=False 或没有 finished 但有 result，继续等待
                if data.get("finished") is False:
                    time.sleep(poll_interval)
                    continue
                # 兼容无 finished 字段的情况（认为有 result 即完成）
                if data.get("result"):
                    return data
            time.sleep(poll_interval)
        except Exception as e:
            print(f"[轮询错误] {task_id}: {e}")
            time.sleep(poll_interval * 2)
    raise TimeoutError(f"任务 {task_id} 超时 ({timeout}s)")

def single_workflow(idx: int) -> Dict:
    """单个任务的完整流程"""
    prompt = f"Write a short fact about the number {idx}."
    start = time.perf_counter()
    try:
        task_id = create_task(prompt, max_tokens=80)
        result = wait_for_result(task_id)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "idx": idx,
            "success": True,
            "task_id": task_id,
            "elapsed_ms": elapsed_ms,
            "speed": result.get("speed", 0),
            "gen_time": result.get("gen_time", 0),
            "prefill_time": result.get("prefill_time", 0),
            "result_preview": result.get("result", "")[:80],
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "idx": idx,
            "success": False,
            "error": str(e),
            "elapsed_ms": elapsed_ms,
        }

def run_batch(concurrency: int, total: int):
    print(f"\n🚀 批量测试开始")
    print(f"   并发数: {concurrency}  总请求: {total}  服务: {BASE_URL}\n")

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(single_workflow, i): i for i in range(total)}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            if res["success"]:
                print(f"✅ [{res['idx']:3d}] 完成  {res['elapsed_ms']:6.0f} ms  speed={res['speed']:5.2f} tok/s  {res['result_preview']}...")
            else:
                print(f"❌ [{res['idx']:3d}] 失败  {res['elapsed_ms']:6.0f} ms  原因: {res.get('error')}")

    # 统计
    success_list = [r for r in results if r["success"]]
    if not success_list:
        print("\n❌ 所有请求均失败！")
        return

    speeds = [r["speed"] for r in success_list if r["speed"] > 0]
    elapsed = [r["elapsed_ms"] for r in success_list]

    print(f"\n📊 测试汇总")
    print(f"   成功: {len(success_list)} / {total}")
    if speeds:
        print(f"   平均生成速度: {sum(speeds)/len(speeds):.2f} tok/s")
        print(f"   最快生成速度: {max(speeds):.2f} tok/s")
        print(f"   最慢生成速度: {min(speeds):.2f} tok/s")
    print(f"   平均端到端耗时: {sum(elapsed)/len(elapsed):.0f} ms")
    print(f"   最小耗时: {min(elapsed):.0f} ms")
    print(f"   最大耗时: {max(elapsed):.0f} ms")

if __name__ == "__main__":
    # 先低并发测试，确保功能正常
    run_batch(concurrency=3, total=5)