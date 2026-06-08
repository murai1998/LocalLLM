from __future__ import annotations

import asyncio
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
async def list_models() -> dict[str, Any]:
    settings = _get_settings()
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
    if _inference_lock is None:
        _inference_lock = asyncio.Semaphore(settings.service.max_concurrent_requests)

    body = await request.json()
    body.setdefault("model", settings.llm.model)

    async with _inference_lock:
        async with httpx.AsyncClient(timeout=settings.llm.timeout_sec) as client:
            upstream = await client.post(
                f"{_llama_base_url()}/v1/chat/completions",
                json=body,
            )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )