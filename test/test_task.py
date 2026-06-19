"""私有 Task API 测试

覆盖端点:
  - POST /v1/tasks/tmp          创建临时任务
  - POST /v1/tasks/create       创建持久化任务
  - GET  /v1/tasks/{id}/get_result  轮询结果
  - POST /v1/tasks/{id}/fork     Fork
  - POST /v1/tasks/{id}/continue  Continue
  - POST /v1/tasks/{id}/stop     停止
  - POST /v1/tasks/{id}/as_template  设为模板
  - POST /v1/tasks/{id}/delete   删除
  - GET  /v1/tasks/list          列表
"""
import pytest

from conftest import BASE_URL, consume_sse, create_task_and_wait


class TestTaskCreate:
    """POST /v1/tasks/tmp"""

    @pytest.mark.asyncio
    async def test_create_returns_task_id(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Hello", "max_tokens": 10, "stream": False},
        )
        assert resp.status == 200
        data = await resp.json()
        assert "task_id" in data
        assert data["task_id"].startswith("TMP_")

    @pytest.mark.asyncio
    async def test_create_with_token_ids(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": [1, 2, 3], "max_tokens": 5, "stream": False},
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_create_with_seed(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Hello", "max_tokens": 5, "stream": False, "seed": 42},
        )
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_create_stream_mode(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Hello", "max_tokens": 10, "stream": True},
        )
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        chunks, done = await consume_sse(resp)
        assert done

    @pytest.mark.asyncio
    async def test_create_max_tokens_zero(self, session, server_alive):
        """max_tokens=0 应立即完成（prefill only）"""
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Hello", "max_tokens": 0, "stream": False},
        )
        assert resp.status == 200


class TestTaskGetResult:
    """GET /v1/tasks/{id}/get_result"""

    @pytest.mark.asyncio
    async def test_poll_returns_finished(self, session, server_alive):
        result = await create_task_and_wait(session, "Say OK", max_tokens=10)
        assert result["finished"] is True
        assert isinstance(result["result"], str)
        assert "gen_time" in result
        assert result["gen_time"] >= 0

    @pytest.mark.asyncio
    async def test_result_has_speed(self, session, server_alive):
        result = await create_task_and_wait(session, "Hello", max_tokens=20)
        assert result["finished"] is True
        assert "speed" in result
        assert result["speed"] > 0

    @pytest.mark.asyncio
    async def test_result_text_non_empty(self, session, server_alive):
        result = await create_task_and_wait(session, "Say hello world", max_tokens=20)
        assert result["finished"] is True
        assert len(result["result"]) > 0


class TestTaskStop:
    """POST /v1/tasks/{id}/stop"""

    @pytest.mark.asyncio
    async def test_stop_running_task(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Write a long essay about AI", "max_tokens": 500, "stream": False},
        )
        task_id = (await resp.json())["task_id"]
        await asyncio.sleep(0.3)  # 等任务开始

        resp = await session.post(f"{BASE_URL}/v1/tasks/{task_id}/stop")
        assert resp.status == 200


class TestTaskList:
    """GET /v1/tasks/list"""

    @pytest.mark.asyncio
    async def test_list_structure(self, session, server_alive):
        resp = await session.get(f"{BASE_URL}/v1/tasks/list")
        assert resp.status == 200
        data = await resp.json()
        assert "cpu_cache_count" in data
        assert "database_count" in data
        assert "total_count" in data

    @pytest.mark.asyncio
    async def test_list_non_negative(self, session, server_alive):
        resp = await session.get(f"{BASE_URL}/v1/tasks/list")
        data = await resp.json()
        assert data["total_count"] >= 0


class TestTaskDelete:
    """POST /v1/tasks/{id}/delete"""

    @pytest.mark.asyncio
    async def test_delete_task(self, session, server_alive):
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/tmp",
            json={"prompt": "Hello", "max_tokens": 5, "stream": False},
        )
        task_id = (await resp.json())["task_id"]

        resp = await session.post(f"{BASE_URL}/v1/tasks/{task_id}/delete")
        assert resp.status == 200


class TestTaskFork:
    """POST /v1/tasks/{id}/fork"""

    @pytest.mark.asyncio
    async def test_fork_task(self, session, server_alive):
        # 先创建一个持久化任务
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/create",
            json={"prompt": "Hello", "max_tokens": 10, "stream": False},
        )
        assert resp.status == 200
        task_id = (await resp.json())["task_id"]

        # fork 需要传 TaskUpdate body
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/{task_id}/fork",
            json={"max_tokens": 10, "stream": False},
        )
        assert resp.status == 200
        data = await resp.json()
        assert "task_id" in data


class TestTaskContinue:
    """POST /v1/tasks/{id}/continue"""

    @pytest.mark.asyncio
    async def test_continue_task(self, session, server_alive):
        # 先创建一个持久化任务
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/create",
            json={"prompt": "Hello", "max_tokens": 10, "stream": False},
        )
        assert resp.status == 200
        task_id = (await resp.json())["task_id"]

        # continue 需要传 TaskUpdate body
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/{task_id}/continue",
            json={"max_tokens": 10, "stream": False},
        )
        assert resp.status == 200


class TestTaskAsTemplate:
    """POST /v1/tasks/{id}/as_template"""

    @pytest.mark.asyncio
    async def test_as_template(self, session, server_alive):
        # 先创建一个持久化任务
        resp = await session.post(
            f"{BASE_URL}/v1/tasks/create",
            json={"prompt": "Hello", "max_tokens": 10, "stream": False},
        )
        assert resp.status == 200
        task_id = (await resp.json())["task_id"]

        resp = await session.post(f"{BASE_URL}/v1/tasks/{task_id}/as_template")
        assert resp.status == 200


# 需要导入 asyncio
import asyncio
