from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient


def test_health_reports_gateway_status():
    with patch("localllm.service.app.LlamaServerManager") as manager_cls:
        manager = MagicMock()
        manager.is_ready.return_value = True
        manager_cls.shared.return_value = manager

        from localllm.service.app import app

        client = TestClient(app)
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["inference_ready"] is True
    assert payload["provider"] == "local"


def test_chat_completions_proxies_to_llama_server():
    with patch("localllm.service.app.LlamaServerManager"):
        from localllm.service.app import app

        upstream = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
            },
            request=httpx.Request("POST", "http://test"),
        )

        with patch("httpx.AsyncClient") as client_cls:
            async_client = AsyncMock()
            async_client.post = AsyncMock(return_value=upstream)
            client_cls.return_value.__aenter__ = AsyncMock(return_value=async_client)
            client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"