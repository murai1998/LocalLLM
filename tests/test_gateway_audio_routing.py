import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from localllm.config import AppSettings

app_module = importlib.import_module("localllm.service.app")

AUDIO_MESSAGE = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Transcribe."},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    ],
}
TEXT_MESSAGE = {"role": "user", "content": "hi"}


@pytest.fixture
def gateway():
    def make(model_name: str):
        app_module._settings = AppSettings(model={"name": model_name})
        app_module._inference_lock = None
        app_module._audio_lock = None
        return app_module

    with patch("localllm.service.app.LlamaServerManager") as manager_cls:
        manager = MagicMock()
        manager.is_ready.return_value = True
        manager_cls.shared.return_value = manager
        yield make
    app_module._settings = None
    app_module._inference_lock = None
    app_module._audio_lock = None


def _post(module, message):
    upstream = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "ok"}}]},
        request=httpx.Request("POST", "http://test"),
    )
    with patch("httpx.AsyncClient") as client_cls:
        async_client = AsyncMock()
        async_client.post = AsyncMock(return_value=upstream)
        client_cls.return_value.__aenter__ = AsyncMock(return_value=async_client)
        client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        with TestClient(module.app) as client:
            response = client.post("/v1/chat/completions", json={"messages": [message]})
    assert response.status_code == 200
    call = async_client.post.call_args
    return call.args[0], call.kwargs["json"]


def test_audio_request_goes_to_sidecar_when_model_lacks_audio(gateway):
    module = gateway("gemma-4-31b")
    url, body = _post(module, AUDIO_MESSAGE)
    assert url.startswith("http://127.0.0.1:8081/")
    assert body["model"] == "gemma-4-12b-it"


def test_text_request_stays_on_primary_model(gateway):
    module = gateway("gemma-4-31b")
    url, body = _post(module, TEXT_MESSAGE)
    assert url.startswith("http://127.0.0.1:8080/")
    assert body["model"] == "gemma-4-31b-it"


def test_audio_stays_on_primary_when_it_has_audio(gateway):
    module = gateway("gemma-4-12b")
    url, _ = _post(module, AUDIO_MESSAGE)
    assert url.startswith("http://127.0.0.1:8080/")


def test_health_reports_primary_and_audio_models(gateway):
    module = gateway("qwen3.8-27b")
    with TestClient(module.app) as client:
        payload = client.get("/health").json()
    assert payload["model"] == "qwen3.8-27b"
    assert payload["audio"]["model"] == "gemma-4-12b-it"
    assert payload["inference_ready"] is True
