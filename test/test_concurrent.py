"""高并发与 SSE 关闭回归测试

覆盖场景:
  - 多并发 chat 请求
  - 多并发 FIM 请求
  - 混合并发（chat + FIM）
  - SSE 流式连接正常关闭（之前的 bug 回归）
"""
import asyncio

import pytest

from conftest import BASE_URL, consume_sse


class TestConcurrentChat:
    """并发 chat 请求"""

    @pytest.mark.asyncio
    async def test_10_concurrent(self, session, server_alive):
        await self._run(session, 10)

    @pytest.mark.asyncio
    async def test_50_concurrent(self, session, server_alive):
        await self._run(session, 50)

    @pytest.mark.asyncio
    async def test_100_concurrent(self, session, server_alive):
        await self._run(session, 100)

    @staticmethod
    async def _run(session, count):
        tasks = [_single_chat(session, i) for i in range(count)]
        results = await asyncio.gather(*tasks)
        failures = [r for r in results if not r]
        assert len(failures) == 0, f"{len(failures)}/{count} chat requests failed"


class TestConcurrentFIM:
    """并发 FIM 请求"""

    @pytest.mark.asyncio
    async def test_10_concurrent(self, session, server_alive):
        await self._run(session, 10)

    @pytest.mark.asyncio
    async def test_50_concurrent(self, session, server_alive):
        await self._run(session, 50)

    @staticmethod
    async def _run(session, count):
        tasks = [_single_fim(session, i) for i in range(count)]
        results = await asyncio.gather(*tasks)
        failures = [r for r in results if not r]
        assert len(failures) == 0, f"{len(failures)}/{count} FIM requests failed"


class TestConcurrentMixed:
    """混合并发（chat + FIM）"""

    @pytest.mark.asyncio
    async def test_20_mixed(self, session, server_alive):
        tasks = []
        for i in range(10):
            tasks.append(_single_chat(session, i))
            tasks.append(_single_fim(session, i))
        results = await asyncio.gather(*tasks)
        failures = [r for r in results if not r]
        assert len(failures) == 0, f"{len(failures)}/20 mixed requests failed"


class TestSSECloseRegression:
    """SSE 流式连接正常关闭 — 回归测试

    之前的 bug：max_tokens 恰好等于 buffer_size 的整数倍时，
    _collect() 未调用 task.finish()，导致 SSE 连接挂住。
    """

    @pytest.mark.asyncio
    async def test_max_tokens_32(self, session, server_alive):
        """buffer_size=32 边界"""
        await self._sse_close_check(session, 32)

    @pytest.mark.asyncio
    async def test_max_tokens_64(self, session, server_alive):
        """2 * buffer_size"""
        await self._sse_close_check(session, 64)

    @pytest.mark.asyncio
    async def test_max_tokens_8192(self, session, server_alive):
        """之前的 bug 触发点"""
        await self._sse_close_check(session, 8192, timeout=300)

    @staticmethod
    async def _sse_close_check(session, max_tokens, timeout=60):
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": max_tokens,
                "stream": True,
            },
        )
        assert resp.status == 200
        _, done = await consume_sse(resp)
        assert done, f"SSE did not close for max_tokens={max_tokens}"


class TestConcurrentSSE:
    """并发 SSE 流式关闭"""

    @pytest.mark.asyncio
    async def test_10_concurrent_sse_close(self, session, server_alive):
        tasks = [_single_sse(session, i, 64) for i in range(10)]
        results = await asyncio.gather(*tasks)
        failures = [r for r in results if not r]
        assert len(failures) == 0, f"{len(failures)} SSE streams failed to close"


class TestSSEClientDisconnect:
    """SSE 客户端中途断开 — 验证服务端不崩溃"""

    @pytest.mark.asyncio
    async def test_chat_disconnect_early(self, session, server_alive):
        """chat SSE 流式传输中途断开，服务端应正常处理"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Write a long essay about AI"}],
                "max_tokens": 500,
                "stream": True,
            },
        )
        assert resp.status == 200
        # 只读一部分就关闭连接
        buffer = b""
        received = False
        async for data in resp.content.iter_any():
            buffer += data
            if b"data: " in buffer:
                received = True
                break
        assert received, "Should have received at least one SSE chunk"
        # 主动关闭（不消费完流）
        resp.close()
        # 等一下，让服务端处理断开
        await asyncio.sleep(1)
        # 验证服务端仍然存活
        check = await session.get(f"{BASE_URL}/v1/models")
        assert check.status == 200, "Server crashed after client disconnect"

    @pytest.mark.asyncio
    async def test_fim_disconnect_early(self, session, server_alive):
        """FIM SSE 流式传输中途断开，服务端应正常处理"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/fim",
            json={
                "prefix": "def long_function():\n    ",
                "suffix": "\n    return result",
                "max_tokens": 500,
                "stream": True,
            },
        )
        assert resp.status == 200
        received = False
        async for data in resp.content.iter_any():
            if b"data: " in data:
                received = True
                break
        assert received
        resp.close()
        await asyncio.sleep(1)
        check = await session.get(f"{BASE_URL}/v1/models")
        assert check.status == 200, "Server crashed after FIM client disconnect"

    @pytest.mark.asyncio
    async def test_completions_disconnect_early(self, session, server_alive):
        """completions SSE 流式传输中途断开"""
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": "Tell me a long story",
                "max_tokens": 500,
                "stream": True,
            },
        )
        assert resp.status == 200
        received = False
        async for data in resp.content.iter_any():
            if b"data: " in data:
                received = True
                break
        assert received
        resp.close()
        await asyncio.sleep(1)
        check = await session.get(f"{BASE_URL}/v1/models")
        assert check.status == 200, "Server crashed after completions client disconnect"

    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect(self, session, server_alive):
        """快速反复连接-断开，服务端应不崩溃"""
        for _ in range(5):
            resp = await session.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "rwkv",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )
            assert resp.status == 200
            # 立即关闭
            resp.close()
            await asyncio.sleep(0.2)

        # 验证服务端存活
        check = await session.get(f"{BASE_URL}/v1/models")
        assert check.status == 200, "Server crashed after rapid connect/disconnect"

    @pytest.mark.asyncio
    async def test_concurrent_disconnect_5(self, session, server_alive):
        """5 个并发 SSE 同时中途断开"""
        async def disconnect_early(idx):
            try:
                resp = await session.post(
                    f"{BASE_URL}/v1/chat/completions",
                    json={
                        "model": "rwkv",
                        "messages": [{"role": "user", "content": f"Long text {idx}"}],
                        "max_tokens": 200,
                        "stream": True,
                    },
                )
                async for data in resp.content.iter_any():
                    if b"data: " in data:
                        break
                resp.close()
                return True
            except Exception:
                return False

        results = await asyncio.gather(*[disconnect_early(i) for i in range(5)])
        assert all(results), "Some concurrent disconnects failed"
        await asyncio.sleep(1)
        check = await session.get(f"{BASE_URL}/v1/models")
        assert check.status == 200, "Server crashed after concurrent disconnects"


# ============================================================
# 辅助函数
# ============================================================

async def _single_chat(session, idx):
    try:
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": f"Fact {idx}"}],
                "max_tokens": 10,
                "stream": False,
            },
        )
        return resp.status == 200
    except Exception:
        return False


async def _single_fim(session, idx):
    try:
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/fim",
            json={
                "prefix": f"def f{idx}():\n    return ",
                "suffix": "\n# end",
                "max_tokens": 10,
                "stream": False,
            },
        )
        return resp.status == 200
    except Exception:
        return False


async def _single_sse(session, idx, max_tokens):
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
        _, done = await consume_sse(resp)
        return done
    except Exception:
        return False
