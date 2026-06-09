#!/usr/bin/env python3
"""
异步批量测试（修复版）- 支持高并发，能处理响应中的控制字符
"""

import asyncio
import aiohttp
import time
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class TaskResult:
    idx: int
    success: bool
    elapsed_ms: float
    speed: float = 0.0
    result_preview: str = ""
    error: str = ""

class AsyncBatchTester:
    def __init__(self, base_url: str, concurrency: int, total: int, timeout: float = 180):
        self.base_url = base_url
        self.concurrency = concurrency
        self.total = total
        self.timeout = timeout
        self.poll_interval = 1.0

    async def create_task(self, session: aiohttp.ClientSession, idx: int) -> Optional[str]:
        url = f"{self.base_url}/v1/tasks/create"
        payload = {
            "prompt": f"Tell a short fact about {idx}.",
            "max_tokens": 80,
            "temperature": 0.7
        }
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("task_id") or data.get("id")
                else:
                    print(f"创建失败 {idx}: HTTP {resp.status}")
                    return None
        except Exception as e:
            print(f"创建异常 {idx}: {e}")
            return None

    async def wait_result(self, session: aiohttp.ClientSession, task_id: str, idx: int) -> Optional[Dict]:
        url = f"{self.base_url}/v1/tasks/{task_id}/get_result"
        start = time.time()
        while time.time() - start < self.timeout:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("finished") is True:
                            return data
                        elif data.get("finished") is False:
                            await asyncio.sleep(self.poll_interval)
                            continue
                        else:
                            # 无 finished 字段，但有 result 则认为完成
                            if data.get("result"):
                                return data
                            await asyncio.sleep(self.poll_interval)
                    else:
                        await asyncio.sleep(self.poll_interval)
            except Exception as e:
                print(f"轮询异常 {task_id}: {e}")
                await asyncio.sleep(self.poll_interval * 2)
        return None

    async def single_workflow(self, session: aiohttp.ClientSession, idx: int) -> TaskResult:
        start = time.perf_counter()
        task_id = await self.create_task(session, idx)
        if not task_id:
            elapsed = (time.perf_counter() - start) * 1000
            return TaskResult(idx=idx, success=False, elapsed_ms=elapsed, error="创建任务失败")

        result = await self.wait_result(session, task_id, idx)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if result:
            speed = result.get("speed", 0.0)
            preview = result.get("result", "")[:80]
            return TaskResult(idx=idx, success=True, elapsed_ms=elapsed_ms,
                              speed=speed, result_preview=preview)
        else:
            return TaskResult(idx=idx, success=False, elapsed_ms=elapsed_ms,
                              error=f"等待超时 ({self.timeout}s)")

    async def run(self):
        print(f"\n⚡ 异步批量测试 (并发={self.concurrency}, 总数={self.total}, 超时={self.timeout}s)")
        sem = asyncio.Semaphore(self.concurrency)

        async def limited_work(session, idx):
            async with sem:
                return await self.single_workflow(session, idx)

        async with aiohttp.ClientSession() as session:
            tasks = [limited_work(session, i) for i in range(self.total)]
            results = await asyncio.gather(*tasks)

        # 统计
        success = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n📊 结果汇总")
        print(f"   成功: {len(success)}/{self.total}")
        if success:
            speeds = [r.speed for r in success if r.speed > 0]
            elapsed = [r.elapsed_ms for r in success]
            if speeds:
                print(f"   平均生成速度: {sum(speeds)/len(speeds):.2f} tok/s")
                print(f"   最快: {max(speeds):.2f} tok/s  最慢: {min(speeds):.2f} tok/s")
            print(f"   平均端到端耗时: {sum(elapsed)/len(elapsed):.0f} ms")
        if failed:
            print(f"   失败: {len(failed)}")
            for f in failed[:5]:
                print(f"     - 任务 {f.idx}: {f.error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    tester = AsyncBatchTester(args.base_url, args.concurrency, args.total, args.timeout)
    asyncio.run(tester.run())