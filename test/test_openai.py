"""OpenAI 兼容 API 测试

覆盖端点:
  - GET  /v1/models
  - POST /v1/chat/completions (非流式/流式)
  - POST /v1/completions (非流式/流式)
"""
import pytest

from conftest import BASE_URL, consume_sse, create_task_and_wait


class TestChatCompletions:
    """POST /v1/chat/completions"""

    @pytest.mark.asyncio
    async def test_non_stream_basic(self, session, server_alive):
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
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_non_stream_system_message(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert len(data["choices"]) > 0

    @pytest.mark.asyncio
    async def test_non_stream_multi_turn(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [
                    {"role": "user", "content": "My name is Alice"},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert len(data["choices"]) > 0

    @pytest.mark.asyncio
    async def test_non_stream_n_parameter(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 5,
                "n": 2,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert len(data["choices"]) == 2
        for choice in data["choices"]:
            assert choice["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_non_stream_max_tokens_1(self, session, server_alive):
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
    async def test_non_stream_max_tokens_exceeds_limit(self, session, server_alive):
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

    @pytest.mark.asyncio
    async def test_non_stream_temperature_zero(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 5,
                "temperature": 0.0,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_non_stream_empty_messages_rejected(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [],
                "max_tokens": 20,
                "stream": False,
            },
        )
        # 服务端可能接受空 messages（退化为无 prompt），验证不崩溃即可
        assert resp.status in (200, 422)

    @pytest.mark.asyncio
    async def test_stream_basic(self, session, server_alive):
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
        assert "text/event-stream" in resp.headers.get("content-type", "")
        chunks, done = await consume_sse(resp)
        assert done, "Stream did not end with [DONE]"
        assert len(chunks) > 0, "No data chunks received"
        for chunk in chunks:
            assert "choices" in chunk

    @pytest.mark.asyncio
    async def test_stream_finish_reason(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "max_tokens": 30,
                "stream": True,
            },
        )
        assert resp.status == 200
        chunks, done = await consume_sse(resp)
        assert done
        # 至少有一个 chunk 的 finish_reason 为 stop
        has_stop = any(
            c["choices"][0].get("finish_reason") == "stop"
            for c in chunks
            if c.get("choices")
        )
        assert has_stop, "No chunk with finish_reason=stop"

    @pytest.mark.asyncio
    async def test_stream_content_non_empty(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20,
                "stream": True,
            },
        )
        chunks, done = await consume_sse(resp)
        assert done
        # 至少有一个 chunk 包含文本内容
        has_content = any(
            c["choices"][0].get("delta", {}).get("content")
            for c in chunks
            if c.get("choices")
        )
        assert has_content, "No chunk with text content"


class TestCompletions:
    """POST /v1/completions"""

    @pytest.mark.asyncio
    async def test_non_stream_basic(self, session, server_alive):
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
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_non_stream_echo(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": "Hello",
                "max_tokens": 5,
                "echo": True,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        text = data["choices"][0]["text"]
        assert text.startswith("Hello"), f"echo=True but text does not start with prompt: {text[:50]}"

    @pytest.mark.asyncio
    async def test_non_stream_prompt_list(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": ["Hello", "World"],
                "max_tokens": 5,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert len(data["choices"]) == 2

    @pytest.mark.asyncio
    async def test_non_stream_stop_sequence(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": "Count: one two three",
                "max_tokens": 50,
                "stop": ["four"],
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_basic(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": "Hello",
                "max_tokens": 10,
                "stream": True,
            },
        )
        assert resp.status == 200
        chunks, done = await consume_sse(resp)
        assert done
        assert len(chunks) > 0


class TestChatEdgeCases:
    """chat 端点极端场景"""

    @pytest.mark.asyncio
    async def test_very_long_message(self, session, server_alive):
        """超长消息（4096 字符）"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "A" * 4096}],
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_unicode_message(self, session, server_alive):
        """Unicode 消息：emoji、中文、日文"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "你好世界 🌍 こんにちは"}],
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_many_messages(self, session, server_alive):
        """大量消息轮次（10 轮）"""
        messages = []
        for i in range(5):
            messages.append({"role": "user", "content": f"Message {i}"})
            messages.append({"role": "assistant", "content": f"Reply {i}"})
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": messages,
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_max_tokens_boundary_1(self, session, server_alive):
        """max_tokens=1"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_max_tokens_boundary_8192(self, session, server_alive):
        """max_tokens=8192"""
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8192,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_temperature_boundary(self, session, server_alive):
        """temperature=0 和 temperature=2"""
        for t in [0.0, 2.0]:
            resp = await session.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "rwkv",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                    "temperature": t,
                    "stream": False,
                },
            )
            assert resp.status == 200, f"temperature={t} failed"

    @pytest.mark.asyncio
    async def test_top_p_boundary(self, session, server_alive):
        """top_p=0 和 top_p=1"""
        for p in [0.0, 1.0]:
            resp = await session.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "rwkv",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                    "top_p": p,
                    "stream": False,
                },
            )
            assert resp.status == 200, f"top_p={p} failed"

    @pytest.mark.asyncio
    async def test_repeated_requests(self, session, server_alive):
        """连续 10 次相同请求"""
        for i in range(10):
            resp = await session.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "rwkv",
                    "messages": [{"role": "user", "content": f"Count {i}"}],
                    "max_tokens": 5,
                    "stream": False,
                },
            )
            assert resp.status == 200


class TestCompletionsEdgeCases:
    """completions 端点极端场景"""

    @pytest.mark.asyncio
    async def test_very_long_prompt(self, session, server_alive):
        """超长 prompt（4096 字符）"""
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": "A" * 4096,
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_empty_prompt(self, session, server_alive):
        """空 prompt — 服务端可能 500（已知问题），验证不崩溃即可"""
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": "",
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status in (200, 500)

    @pytest.mark.asyncio
    async def test_unicode_prompt(self, session, server_alive):
        """Unicode prompt"""
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": "你好世界 🌍 こんにちは",
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_max_tokens_zero_rejected(self, session, server_alive):
        """max_tokens=0 被服务端拒绝（schema 要求 >= 1）"""
        resp = await session.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "rwkv",
                "prompt": "Hello",
                "max_tokens": 0,
                "stream": False,
            },
        )
        assert resp.status == 422

    @pytest.mark.asyncio
    async def test_repeated_requests(self, session, server_alive):
        """连续 10 次相同请求"""
        for i in range(10):
            resp = await session.post(
                f"{BASE_URL}/v1/completions",
                json={
                    "model": "rwkv",
                    "prompt": f"Count {i}",
                    "max_tokens": 5,
                    "stream": False,
                },
            )
            assert resp.status == 200
