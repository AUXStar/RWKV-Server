"""共享 fixture 和工具函数"""
import asyncio
import json
import time

import aiohttp
import pytest
import pytest_asyncio

BASE_URL = "http://localhost:8000"
TIMEOUT = 60


@pytest_asyncio.fixture
async def session():
    """共享 aiohttp session"""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=TIMEOUT)
    ) as s:
        yield s


@pytest_asyncio.fixture
async def server_alive(session):
    """检查服务器是否在线，跳过测试如果离线"""
    try:
        resp = await session.get(f"{BASE_URL}/v1/models", timeout=aiohttp.ClientTimeout(total=5))
        if resp.status != 200:
            pytest.skip("Server not ready")
    except (aiohttp.ClientError, asyncio.TimeoutError):
        pytest.skip("Server not reachable")


# ============================================================
# SSE 工具函数
# ============================================================

async def consume_sse(resp) -> tuple[list, bool]:
    """消费 SSE 流，返回 (chunks, done)

    chunks: list[dict] — 解析后的 JSON 数据
    done: bool — 是否收到 [DONE]
    """
    chunks = []
    done = False
    buffer = b""
    async for data in resp.content.iter_any():
        buffer += data
        while b"\n\n" in buffer:
            line, buffer = buffer.split(b"\n\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith(b"data: "):
                payload = line[6:]
                if payload == b"[DONE]":
                    done = True
                else:
                    try:
                        chunks.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
    return chunks, done


async def create_task_and_wait(session, prompt, max_tokens=20, timeout=30):
    """创建非流式任务并等待完成，返回 TaskResponseModel"""
    resp = await session.post(
        f"{BASE_URL}/v1/tasks/tmp",
        json={"prompt": prompt, "max_tokens": max_tokens, "stream": False},
    )
    assert resp.status == 200
    data = await resp.json()
    task_id = data["task_id"]

    for _ in range(int(timeout / 0.5)):
        await asyncio.sleep(0.5)
        resp = await session.get(f"{BASE_URL}/v1/tasks/{task_id}/get_result")
        result = await resp.json()
        if result.get("finished"):
            return result

    pytest.fail(f"Task {task_id} did not finish in {timeout}s")
