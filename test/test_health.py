"""健康检查与模型信息"""
import pytest

from conftest import BASE_URL


class TestHealth:
    @pytest.mark.asyncio
    async def test_models_endpoint(self, session, server_alive):
        resp = await session.get(f"{BASE_URL}/v1/models")
        assert resp.status == 200
        data = await resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"

    @pytest.mark.asyncio
    async def test_models_response_structure(self, session, server_alive):
        resp = await session.get(f"{BASE_URL}/v1/models")
        data = await resp.json()
        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert "created" in model
            assert "owned_by" in model
