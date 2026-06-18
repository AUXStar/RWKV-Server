"""
RWKV-Server 功能测试与高并发测试
pytest + pytest-asyncio

用法:
  pytest test/ -v                    # 运行所有测试
  pytest test/ -v -k "fim"          # 只运行 FIM 测试
  pytest test/ -v -k "concurrent"   # 只运行高并发测试
  pytest test/ -v -k "sse"          # 只运行 SSE 测试

注意: 运行前需启动 RWKV-Server
"""
import asyncio
import json
import time

import aiohttp
import pytest
import pytest_asyncio


BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 60


@pytest_asyncio.fixture
async def session():
    """共享的 aiohttp session"""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
    ) as s:
        yield s


# ============================================================
# 基础健康检查
# ============================================================

class TestHealth:
    """服务健康检查"""

    @pytest.mark.asyncio
    async def test_models_endpoint(self, session):
        resp = await session.get(f"{BASE_URL}/v1/models")
        assert resp.status == 200
        data = await resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert "id" in data["data"][0]


# ============================================================
# Task API 测试
# ============================================================

class TestTaskAPI:
    """Task CRUD API 功能测试"""

    @pytest.mark.asyncio
    async def test_create_task(self, session):
        """创建任务应返回 task_id"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Hello", "max_tokens": 10, "stream": False},
        )
        assert resp.status == 200
        data = await resp.json()
        assert "task_id" in data
        assert data["task_id"].startswith("TMP_")

    @pytest.mark.asyncio
    async def test_create_and_poll_result(self, session):
        """创建任务后轮询，应能获取生成结果"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Say hi", "max_tokens": 20, "stream": False},
        )
        data = await resp.json()
        task_id = data["task_id"]

        # 轮询等待完成
        for _ in range(30):
            await asyncio.sleep(0.5)
            resp = await session.get(
                f"{BASE_URL}/v1/tasks/{task_id}/get_result"
            )
            result = await resp.json()
            if result.get("finished"):
                break
        else:
            pytest.fail("Task did not finish in time")

        assert result["finished"] is True
        assert isinstance(result["result"], str)

    @pytest.mark.asyncio
    async def test_stream_task(self, session):
        """流式任务应返回 SSE 事件并以 [DONE] 结尾"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Count to 5", "max_tokens": 30, "stream": True},
        )
        assert resp.status == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

        chunks = []
        done = False
        buffer = b""
        async for data in resp.content.iter_any():
            buffer += data
            while b"\n\n" in buffer:
                line, buffer = buffer.split(b"\n\n", 1)
                line = line.strip()
                if line.startswith(b"data: "):
                    payload = line[6:]
                    if payload == b"[DONE]":
                        done = True
                    else:
                        chunks.append(json.loads(payload))

        assert done, "Stream did not end with [DONE]"
        assert len(chunks) > 0, "No data chunks received"

    @pytest.mark.asyncio
    async def test_stop_task(self, session):
        """停止任务应成功"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Write a long essay", "max_tokens": 500, "stream": False},
        )
        data = await resp.json()
        task_id = data["task_id"]

        # 等一下让任务开始
        await asyncio.sleep(0.3)

        resp = await session.post(f"{BASE_URL}/v1/tasks/{task_id}/stop")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_list_tasks(self, session):
        """任务列表应返回有效结构"""
        resp = await session.get(f"{BASE_URL}/v1/tasks/list")
        assert resp.status == 200
        data = await resp.json()
        assert "cpu_cache_count" in data
        assert "database_count" in data
        assert "total_count" in data


# ============================================================
# OpenAI 兼容 API 测试
# ============================================================

class TestOpenAIAPI:
    """OpenAI 兼容 API 功能测试"""

    @pytest.mark.asyncio
    async def test_chat_completions_non_stream(self, session):
        """非流式 chat completions"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_chat_completions_stream(self, session):
        """流式 chat completions 应以 [DONE] 结尾"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 20,
                "stream": True,
            },
        )
        assert resp.status == 200

        done = False
        buffer = b""
        async for data in resp.content.iter_any():
            buffer += data
            while b"\n\n" in buffer:
                line, buffer = buffer.split(b"\n\n", 1)
                line = line.strip()
                if line.startswith(b"data: "):
                    if line[6:] == b"[DONE]":
                        done = True
                    else:
                        chunk = json.loads(line[6:])
                        assert "choices" in chunk

        assert done, "Stream did not end with [DONE]"

    @pytest.mark.asyncio
    async def test_completions_non_stream(self, session):
        """非流式 text completions"""
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_max_tokens_boundary(self, session):
        """max_tokens=1 应正常完成"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_max_tokens_exceeds_limit(self, session):
        """max_tokens 超过上限应返回 422"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 99999,
                "stream": False,
            },
        )
        assert resp.status == 422


# ============================================================
# FIM (Fill In Middle) 测试
# ============================================================

class TestFIM:
    """FIM 功能测试"""

    @pytest.mark.asyncio
    async def test_fim_non_stream(self, session):
        """非流式 FIM 应返回 task_id"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/fim",
            json={
                "prefix": "def add(a, b):\n    return ",
                "suffix": "\n    # end",
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert "task_id" in data

    @pytest.mark.asyncio
    async def test_fim_poll_result(self, session):
        """FIM 非流式轮询应获取生成结果"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/fim",
            json={
                "prefix": "def hello():\n    print(",
                "suffix": ")\n    return True",
                "max_tokens": 30,
                "stream": False,
            },
        )
        task_id = (await resp.json())["task_id"]

        for _ in range(30):
            await asyncio.sleep(0.5)
            resp = await session.get(
                f"{BASE_URL}/v1/tasks/{task_id}/get_result"
            )
            result = await resp.json()
            if result.get("finished"):
                break
        else:
            pytest.fail("FIM task did not finish in time")

        assert result["finished"] is True

    @pytest.mark.asyncio
    async def test_fim_stream(self, session):
        """流式 FIM 应返回 SSE 事件"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/fim",
            json={
                "prefix": "The quick brown fox ",
                "suffix": " jumped over the lazy dog.",
                "max_tokens": 30,
                "stream": True,
            },
        )
        assert resp.status == 200

        done = False
        chunks = []
        buffer = b""
        async for data in resp.content.iter_any():
            buffer += data
            while b"\n\n" in buffer:
                line, buffer = buffer.split(b"\n\n", 1)
                line = line.strip()
                if line.startswith(b"data: "):
                    payload = line[6:]
                    if payload == b"[DONE]":
                        done = True
                    else:
                        chunks.append(json.loads(payload))

        assert done, "FIM stream did not end with [DONE]"
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_fim_empty_suffix(self, session):
        """空 suffix 的 FIM 应正常工作"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/fim",
            json={
                "prefix": "Once upon a time,",
                "suffix": "",
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_fim_prompt_format(self, session):
        """验证 FIM prompt 构造格式正确"""
        # 通过非流式创建，然后检查 test.log 中的 prompt
        # 这里只验证 API 接受参数，不验证内部 prompt
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/fim",
            json={
                "prefix": "abc",
                "suffix": "xyz",
                "max_tokens": 5,
                "stream": False,
            },
        )
        assert resp.status == 200


# ============================================================
# SSE 连接关闭测试（之前的 bug 回归测试）
# ============================================================

class TestSSEClose:
    """SSE 流式连接正常关闭的回归测试"""

    @pytest.mark.asyncio
    async def test_sse_close_max_tokens_32(self, session):
        """max_tokens=32（buffer_size）应正常关闭"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 32,
                "stream": True,
            },
        )
        done = await self._consume_sse(resp)
        assert done

    @pytest.mark.asyncio
    async def test_sse_close_max_tokens_8192(self, session):
        """max_tokens=8192 应正常关闭（之前的 bug 场景）"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8192,
                "stream": True,
            },
        )
        done = await self._consume_sse(resp, timeout=300)
        assert done

    @staticmethod
    async def _consume_sse(resp, timeout=60):
        """消费 SSE 流，返回是否收到 [DONE]"""
        done = False
        buffer = b""
        start = time.time()
        async for data in resp.content.iter_any():
            buffer += data
            while b"\n\n" in buffer:
                line, buffer = buffer.split(b"\n\n", 1)
                line = line.strip()
                if line.startswith(b"data: "):
                    if line[6:] == b"[DONE]":
                        done = True
            if time.time() - start > timeout:
                break
        return done


# ============================================================
# 高并发测试
# ============================================================

class TestHighConcurrency:
    """高并发功能测试"""

    @pytest.mark.asyncio
    async def test_concurrent_chat_10(self, session):
        """10 并发 chat 请求应全部成功"""
        await self._run_concurrent_chat(session, count=10, max_tokens=20)

    @pytest.mark.asyncio
    async def test_concurrent_chat_50(self, session):
        """50 并发 chat 请求应全部成功"""
        await self._run_concurrent_chat(session, count=50, max_tokens=20)

    @pytest.mark.asyncio
    async def test_concurrent_chat_100(self, session):
        """100 并发 chat 请求应全部成功"""
        await self._run_concurrent_chat(session, count=100, max_tokens=20)

    @pytest.mark.asyncio
    async def test_concurrent_fim_10(self, session):
        """10 并发 FIM 请求应全部成功"""
        await self._run_concurrent_fim(session, count=10, max_tokens=20)

    @pytest.mark.asyncio
    async def test_concurrent_fim_50(self, session):
        """50 并发 FIM 请求应全部成功"""
        await self._run_concurrent_fim(session, count=50, max_tokens=20)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_20(self, session):
        """20 并发混合请求（chat + FIM）应全部成功"""
        tasks = []
        for i in range(10):
            tasks.append(self._single_chat(session, i, max_tokens=20))
            tasks.append(self._single_fim(session, i, max_tokens=20))

        results = await asyncio.gather(*tasks)
        failures = [r for r in results if not r]
        assert len(failures) == 0, f"{len(failures)} requests failed"

    @pytest.mark.asyncio
    async def test_concurrent_sse_close_10(self, session):
        """10 并发 SSE 流式请求应全部正常关闭"""
        tasks = []
        for i in range(10):
            tasks.append(self._single_sse(session, i, max_tokens=64))

        results = await asyncio.gather(*tasks)
        failures = [r for r in results if not r]
        assert len(failures) == 0, f"{len(failures)} SSE streams failed to close"

    @staticmethod
    async def _run_concurrent_chat(session, count, max_tokens):
        tasks = [TestHighConcurrency._single_chat(session, i, max_tokens) for i in range(count)]
        results = await asyncio.gather(*tasks)
        failures = [r for r in results if not r]
        assert len(failures) == 0, f"{len(failures)}/{count} chat requests failed"

    @staticmethod
    async def _run_concurrent_fim(session, count, max_tokens):
        tasks = [TestHighConcurrency._single_fim(session, i, max_tokens) for i in range(count)]
        results = await asyncio.gather(*tasks)
        failures = [r for r in results if not r]
        assert len(failures) == 0, f"{len(failures)}/{count} FIM requests failed"

    @staticmethod
    async def _single_chat(session, idx, max_tokens):
        """单个 chat 请求（非流式）"""
        try:
            resp = await session.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "rwkv",
                    "messages": [{"role": "user", "content": f"Fact about {idx}"}],
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            return resp.status == 200
        except Exception:
            return False

    @staticmethod
    async def _single_fim(session, idx, max_tokens):
        """单个 FIM 请求（非流式）"""
        try:
            resp = await session.post(
                f"{BASE_URL}/v1/tasks/fim",
                json={
                    "prefix": f"def func_{idx}():\n    return ",
                    "suffix": "\n    # end",
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            return resp.status == 200
        except Exception:
            return False

    @staticmethod
    async def _single_sse(session, idx, max_tokens):
        """单个 SSE 流式请求，验证正常关闭"""
        try:
            resp = await session.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "rwkv",
                    "messages": [{"role": "user", "content": f"Count {idx}"}],
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            )
            done = False
            buffer = b""
            async for data in resp.content.iter_any():
                buffer += data
                while b"\n\n" in buffer:
                    line, buffer = buffer.split(b"\n\n", 1)
                    line = line.strip()
                    if line.startswith(b"data: ") and line[6:] == b"[DONE]":
                        done = True
            return done
        except Exception:
            return False
