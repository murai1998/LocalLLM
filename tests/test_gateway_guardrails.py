import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from localllm.config import AppSettings

# Import via importlib to be robust against any future `app` attribute
# shadowing in the `localllm.service` package (see its __init__ note).
app_module = importlib.import_module("localllm.service.app")


@pytest.fixture
def gateway():
    """Gateway app with a mocked llama-server manager and fresh module state."""
    with patch("localllm.service.app.LlamaServerManager") as manager_cls:
        manager = MagicMock()
        manager.is_ready.return_value = True
        manager_cls.shared.return_value = manager

        app_module._settings = AppSettings()
        app_module._inference_lock = None
        yield app_module
        app_module._settings = None
        app_module._inference_lock = None


def _mock_upstream(response=None, error=None):
    client_cls = patch("httpx.AsyncClient")
    mocked = client_cls.start()
    async_client = AsyncMock()
    if error is not None:
        async_client.post = AsyncMock(side_effect=error)
    else:
        async_client.post = AsyncMock(return_value=response)
    mocked.return_value.__aenter__ = AsyncMock(return_value=async_client)
    mocked.return_value.__aexit__ = AsyncMock(return_value=None)
    return client_cls


def test_upstream_down_returns_502(gateway):
    patcher = _mock_upstream(error=httpx.ConnectError("refused"))
    try:
        client = TestClient(gateway.app)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        patcher.stop()

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


def test_upstream_timeout_returns_504(gateway):
    patcher = _mock_upstream(error=httpx.ReadTimeout("slow"))
    try:
        client = TestClient(gateway.app)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        patcher.stop()

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "upstream_timeout"


def test_oversized_body_returns_413(gateway):
    gateway._settings.service.max_request_bytes = 64
    client = TestClient(gateway.app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x" * 1024}]},
    )
    assert response.status_code == 413


def test_invalid_json_returns_400(gateway):
    client = TestClient(gateway.app)
    response = client.post(
        "/v1/chat/completions",
        content=b"this is not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


def test_non_object_json_returns_400(gateway):
    client = TestClient(gateway.app)
    response = client.post(
        "/v1/chat/completions",
        json=["not", "an", "object"],
    )
    assert response.status_code == 400


def test_busy_queue_returns_503(gateway):
    gateway._settings.service.queue_timeout_sec = 0.05
    gateway._inference_lock = asyncio.Semaphore(0)  # all slots taken
    client = TestClient(gateway.app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_busy"


def test_missing_api_key_returns_401(gateway):
    gateway._settings.service.api_key = "sekrit"
    client = TestClient(gateway.app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401


def test_valid_api_key_passes_through(gateway):
    gateway._settings.service.api_key = "sekrit"
    upstream = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "hello"}}]},
        request=httpx.Request("POST", "http://test"),
    )
    patcher = _mock_upstream(response=upstream)
    try:
        client = TestClient(gateway.app)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sekrit"},
        )
    finally:
        patcher.stop()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"


def test_health_does_not_require_api_key(gateway):
    gateway._settings.service.api_key = "sekrit"
    client = TestClient(gateway.app)
    response = client.get("/health")
    assert response.status_code == 200
