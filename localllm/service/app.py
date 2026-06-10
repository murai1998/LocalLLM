from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from localllm.backends.llama_server import LlamaServerManager
from localllm.config import AppSettings, get_settings
from localllm.devices.resolver import detect_platform

_inference_lock: asyncio.Semaphore | None = None
_settings: AppSettings | None = None


def _get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def _llama_base_url() -> str:
    return _get_settings().llama_server.base_url.rstrip("/")


def _error_response(status_code: int, message: str, *, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": status_code}},
    )


def _check_auth(request: Request, settings: AppSettings) -> JSONResponse | None:
    """Optional bearer auth for /v1/* endpoints; disabled when api_key unset."""
    expected = settings.service.api_key
    if not expected:
        return None
    header = request.headers.get("authorization", "")
    if header == f"Bearer {expected}":
        return None
    return _error_response(401, "Invalid or missing API key.", error_type="authentication_error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_lock
    settings = _get_settings()
    _inference_lock = asyncio.Semaphore(settings.service.max_concurrent_requests)

    if settings.service.autostart_llama_server:
        LlamaServerManager.shared(settings).start()

    yield

    LlamaServerManager.shared(settings).stop()


app = FastAPI(
    title="LocalLLM Gateway",
    description="OpenAI-compatible gateway over a single llama-server instance.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = _get_settings()
    manager = LlamaServerManager.shared(settings)
    inference_ready = manager.is_ready()
    return {
        "status": "ok" if inference_ready else "starting",
        "provider": settings.llm.provider,
        "platform": detect_platform(),
        "inference_ready": inference_ready,
        "gateway": settings.service.base_url,
        "inference": settings.llama_server.base_url,
        "model": settings.llm.model,
    }


@app.get("/v1/models")
async def list_models(request: Request) -> Any:
    settings = _get_settings()
    auth_error = _check_auth(request, settings)
    if auth_error is not None:
        return auth_error
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{_llama_base_url()}/v1/models")
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            pass
    return {
        "object": "list",
        "data": [{"id": settings.llm.model, "object": "model"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    global _inference_lock
    settings = _get_settings()

    auth_error = _check_auth(request, settings)
    if auth_error is not None:
        return auth_error

    max_bytes = settings.service.max_request_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        return _error_response(
            413,
            f"Request body exceeds the {max_bytes} byte limit.",
            error_type="invalid_request_error",
        )
    raw = await request.body()
    if len(raw) > max_bytes:
        return _error_response(
            413,
            f"Request body exceeds the {max_bytes} byte limit.",
            error_type="invalid_request_error",
        )

    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _error_response(
            400, "Request body must be valid JSON.", error_type="invalid_request_error"
        )
    if not isinstance(body, dict):
        return _error_response(
            400, "Request body must be a JSON object.", error_type="invalid_request_error"
        )
    body.setdefault("model", settings.llm.model)

    if _inference_lock is None:
        _inference_lock = asyncio.Semaphore(settings.service.max_concurrent_requests)
    lock = _inference_lock

    try:
        await asyncio.wait_for(lock.acquire(), timeout=settings.service.queue_timeout_sec)
    except asyncio.TimeoutError:
        return _error_response(
            503,
            "All inference slots are busy; retry shortly.",
            error_type="server_busy",
        )

    try:
        async with httpx.AsyncClient(timeout=settings.llm.timeout_sec) as client:
            upstream = await client.post(
                f"{_llama_base_url()}/v1/chat/completions",
                json=body,
            )
    except httpx.TimeoutException:
        return _error_response(
            504, "Inference backend timed out.", error_type="upstream_timeout"
        )
    except httpx.HTTPError as exc:
        return _error_response(
            502,
            f"Inference backend is unavailable: {exc.__class__.__name__}",
            error_type="upstream_error",
        )
    finally:
        lock.release()

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
