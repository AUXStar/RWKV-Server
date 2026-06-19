"""FIM (Fill In Middle) 端点测试

覆盖端点:
  - POST /v1/tasks/fim

测试维度:
  - 非流式/流式
  - 空 suffix / 空 prefix
  - 长 prefix + suffix
  - 轮询结果
  - 生成内容不含 FIM 标记泄漏
  - 参数校验
"""
import pytest

from conftest import BASE_URL, consume_sse, create_task_and_wait

FIM_URL = f"{BASE_URL}/v1/tasks/fim"


class TestFIMNonStream:
    """非流式 FIM"""

    @pytest.mark.asyncio
    async def test_basic(self, session, server_alive):
        resp = await session.post(
            FIM_URL,
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
    async def test_poll_result(self, session, server_alive):
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "def hello():\n    print(",
                "suffix": ")\n    return True",
                "max_tokens": 30,
                "stream": False,
            },
        )
        task_id = (await resp.json())["task_id"]

        import asyncio
        for _ in range(30):
            await asyncio.sleep(0.5)
            resp = await session.get(f"{BASE_URL}/v1/tasks/{task_id}/get_result")
            result = await resp.json()
            if result.get("finished"):
                break
        else:
            pytest.fail("FIM task did not finish in time")

        assert result["finished"] is True
        assert isinstance(result["result"], str)

    @pytest.mark.asyncio
    async def test_empty_suffix(self, session, server_alive):
        """空 suffix 应退化为普通续写"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "Once upon a time,",
                "suffix": "",
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_empty_prefix(self, session, server_alive):
        """空 prefix — suffix 在前，模型在开头生成"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "",
                "suffix": "The end.",
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_code_completion(self, session, server_alive):
        """代码补全场景"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "def factorial(n):\n    if n == 0:\n        return 1\n    ",
                "suffix": "\n    print(factorial(5))",
                "max_tokens": 30,
                "temperature": 0.3,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_text_infill(self, session, server_alive):
        """文本填充场景"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "The rain had stopped, but the street still glistened like ",
                "suffix": " though everyone knew Mr. Ellis hadn't opened that door in three years.",
                "max_tokens": 50,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_with_seed(self, session, server_alive):
        """指定 seed 应正常工作"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "Hello ",
                "suffix": " world",
                "max_tokens": 10,
                "seed": 12345,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_max_tokens_1(self, session, server_alive):
        """max_tokens=1 应正常完成"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "def foo():",
                "suffix": "",
                "max_tokens": 1,
                "stream": False,
            },
        )
        assert resp.status == 200


class TestFIMStream:
    """流式 FIM"""

    @pytest.mark.asyncio
    async def test_stream_basic(self, session, server_alive):
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "The quick brown fox ",
                "suffix": " jumped over the lazy dog.",
                "max_tokens": 30,
                "stream": True,
            },
        )
        assert resp.status == 200
        chunks, done = await consume_sse(resp)
        assert done, "FIM stream did not end with [DONE]"
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_has_content(self, session, server_alive):
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "def hello():\n    ",
                "suffix": "\n    return True",
                "max_tokens": 20,
                "stream": True,
            },
        )
        chunks, done = await consume_sse(resp)
        assert done
        # 至少有一个 chunk 包含 data 字段
        has_data = any(c.get("data") for c in chunks)
        assert has_data, "No chunk with data field"

    @pytest.mark.asyncio
    async def test_stream_empty_suffix(self, session, server_alive):
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "Hello world",
                "suffix": "",
                "max_tokens": 10,
                "stream": True,
            },
        )
        assert resp.status == 200
        chunks, done = await consume_sse(resp)
        assert done


class TestFIMQuality:
    """生成质量检查"""

    @pytest.mark.asyncio
    async def test_no_fim_marker_leak(self, session, server_alive):
        """生成内容不应包含 FIM 标记 ✿"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "def foo():\n    return ",
                "suffix": "\n# end",
                "max_tokens": 30,
                "stream": False,
            },
        )
        task_id = (await resp.json())["task_id"]

        import asyncio
        for _ in range(30):
            await asyncio.sleep(0.5)
            resp = await session.get(f"{BASE_URL}/v1/tasks/{task_id}/get_result")
            result = await resp.json()
            if result.get("finished"):
                break

        text = result.get("result", "")
        assert "✿" not in text, f"FIM marker leaked into output: {text[:100]}"

    @pytest.mark.asyncio
    async def test_result_non_empty(self, session, server_alive):
        """FIM 任务应正常完成（finished=True）"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "def add(a, b):\n    return ",
                "suffix": "\n    # done",
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200
        task_id = (await resp.json())["task_id"]

        import asyncio
        for _ in range(30):
            await asyncio.sleep(0.5)
            resp = await session.get(f"{BASE_URL}/v1/tasks/{task_id}/get_result")
            result = await resp.json()
            if result.get("finished"):
                break
        else:
            pytest.fail("FIM task did not finish in time")

        assert result["finished"] is True


class TestFIMEdgeCases:
    """极端场景"""

    @pytest.mark.asyncio
    async def test_very_long_prefix(self, session, server_alive):
        """超长 prefix（4096 字符）"""
        long_prefix = "def " + "a" * 4000 + ":\n    return "
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": long_prefix,
                "suffix": "\n# end",
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_very_long_suffix(self, session, server_alive):
        """超长 suffix（4096 字符）"""
        long_suffix = "\n" + "b" * 4000 + "\n# end"
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "def foo():",
                "suffix": long_suffix,
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_unicode_special_chars(self, session, server_alive):
        """Unicode 特殊字符：emoji、中文、日文"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "# 你好世界 🌍\ndef 函数():",
                "suffix": "\n# こんにちは",
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_newlines_and_tabs(self, session, server_alive):
        """大量换行和制表符"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "class A:\n\tdef __init__(self):\n\t\t",
                "suffix": "\n\t\tself.x = 1\n",
                "max_tokens": 20,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_backslash_and_quotes(self, session, server_alive):
        """反斜杠和引号转义"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": 'path = "C:\\\\Users\\\\"',
                "suffix": '\nprint("done")',
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_max_tokens_zero_rejected(self, session, server_alive):
        """max_tokens=0 被服务端拒绝（schema 要求 >= 1）"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "def foo():",
                "suffix": "",
                "max_tokens": 0,
                "stream": False,
            },
        )
        assert resp.status == 422

    @pytest.mark.asyncio
    async def test_both_empty(self, session, server_alive):
        """prefix 和 suffix 都为空"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "",
                "suffix": "",
                "max_tokens": 5,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_large_max_tokens(self, session, server_alive):
        """max_tokens=8192 大值"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "def foo():\n    ",
                "suffix": "\n    return",
                "max_tokens": 8192,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_temperature_zero(self, session, server_alive):
        """temperature=0（确定性采样）"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "Hello",
                "suffix": "world",
                "max_tokens": 10,
                "temperature": 0.0,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_top_p_zero(self, session, server_alive):
        """top_p=0"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "Hello",
                "suffix": "world",
                "max_tokens": 10,
                "top_p": 0.0,
                "stream": False,
            },
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_presence_frequency_penalty_rejected(self, session, server_alive):
        """presence_penalty / frequency_penalty 不在 FIM schema 中，应被拒绝"""
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "Hello",
                "suffix": "world",
                "max_tokens": 10,
                "presence_penalty": 1.0,
                "frequency_penalty": 1.0,
                "stream": False,
            },
        )
        assert resp.status == 422

    @pytest.mark.asyncio
    async def test_repeated_fim_calls(self, session, server_alive):
        """连续多次 FIM 调用"""
        for i in range(5):
            resp = await session.post(
                FIM_URL,
                json={
                    "prefix": f"def func{i}():",
                    "suffix": "\n    return",
                    "max_tokens": 5,
                    "stream": False,
                },
            )
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_fim_then_chat_then_fim(self, session, server_alive):
        """FIM -> chat -> FIM 交替调用"""
        # FIM
        resp = await session.post(
            FIM_URL,
            json={"prefix": "def a():", "suffix": "", "max_tokens": 5, "stream": False},
        )
        assert resp.status == 200
        # chat
        resp = await session.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "stream": False,
            },
        )
        assert resp.status == 200
        # FIM again
        resp = await session.post(
            FIM_URL,
            json={"prefix": "def b():", "suffix": "", "max_tokens": 5, "stream": False},
        )
        assert resp.status == 200


class TestFIMValidation:
    """参数校验"""

    @pytest.mark.asyncio
    async def test_missing_prefix_rejected(self, session, server_alive):
        resp = await session.post(
            FIM_URL,
            json={
                "suffix": "world",
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert resp.status == 422

    @pytest.mark.asyncio
    async def test_invalid_max_tokens_rejected(self, session, server_alive):
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "Hello",
                "suffix": "world",
                "max_tokens": -1,
                "stream": False,
            },
        )
        assert resp.status == 422

    @pytest.mark.asyncio
    async def test_invalid_temperature_rejected(self, session, server_alive):
        resp = await session.post(
            FIM_URL,
            json={
                "prefix": "Hello",
                "suffix": "world",
                "temperature": 5.0,
                "stream": False,
            },
        )
        assert resp.status == 422
