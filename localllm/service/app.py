from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from localllm.backends.llama_server import LlamaServerManager
from localllm.config import AppSettings, get_settings
from localllm.devices.resolver import detect_platform

_inference_lock: asyncio.Semaphore | None = None
_audio_lock: asyncio.Semaphore | None = None
_settings: AppSettings | None = None


def _get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def _llama_base_url() -> str:
    return _get_settings().llama_server.base_url.rstrip("/")


def _has_audio_input(body: dict[str, Any]) -> bool:
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "input_audio" for part in content
        ):
            return True
    return False


def _audio_manager(settings: AppSettings) -> LlamaServerManager:
    return LlamaServerManager.shared(settings.audio_model_settings(), role="audio")


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
    global _inference_lock, _audio_lock
    settings = _get_settings()
    _inference_lock = asyncio.Semaphore(settings.service.max_concurrent_requests)
    _audio_lock = asyncio.Semaphore(settings.audio_sidecar.max_concurrent_requests)
    sidecar = settings.audio_sidecar_active()

    if settings.service.autostart_llama_server:
        if sidecar:
            # Load both models in parallel; the primary start blocks until ready.
            _audio_manager(settings).start(wait=False)
        LlamaServerManager.shared(settings).start()
        if sidecar:
            _audio_manager(settings).start()

    yield

    LlamaServerManager.shared(settings).stop()
    if sidecar:
        _audio_manager(settings).stop()


app = FastAPI(
    title="LocalLLM Gateway",
    description="OpenAI-compatible gateway over llama-server (main model + optional audio sidecar).",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = _get_settings()
    manager = LlamaServerManager.shared(settings)
    inference_ready = manager.is_ready()
    preset = settings.model.preset()
    audio: dict[str, Any] = {"model": preset.served_name, "ready": inference_ready}
    if settings.audio_sidecar_active():
        audio_settings = settings.audio_model_settings()
        audio = {
            "model": audio_settings.model.preset().served_name,
            "ready": _audio_manager(settings).is_ready(),
            "inference": audio_settings.llama_server.base_url,
        }
    ready = inference_ready and bool(audio["ready"])
    return {
        "status": "ok" if ready else "starting",
        "provider": settings.llm.provider,
        "platform": detect_platform(),
        "inference_ready": ready,
        "gateway": settings.service.base_url,
        "inference": settings.llama_server.base_url,
        "model": preset.served_name,
        "model_label": preset.label,
        "quantization": settings.model.quantization or preset.default_quantization,
        "audio": audio,
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
    global _inference_lock, _audio_lock
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
    if _inference_lock is None:
        _inference_lock = asyncio.Semaphore(settings.service.max_concurrent_requests)
    if _audio_lock is None:
        _audio_lock = asyncio.Semaphore(settings.audio_sidecar.max_concurrent_requests)

    if settings.audio_sidecar_active() and _has_audio_input(body):
        # The main model has no audio encoder: speech goes to the sidecar, on its
        # own slot pool so STT never queues behind long text generations.
        audio_settings = settings.audio_model_settings()
        base_url = audio_settings.llama_server.base_url.rstrip("/")
        body["model"] = audio_settings.model.preset().served_name
        lock = _audio_lock
    else:
        base_url = _llama_base_url()
        body["model"] = settings.model.preset().served_name
        lock = _inference_lock

    try:
        await asyncio.wait_for(lock.acquire(), timeout=settings.service.queue_timeout_sec)
    except asyncio.TimeoutError:
        return _error_response(
            503,
            "All inference slots are busy; retry shortly.",
            error_type="server_busy",
        )

    url = f"{base_url}/v1/chat/completions"

    if body.get("stream"):
        # SSE pass-through: hold the inference slot for the lifetime of the
        # stream and release it when the relay generator closes.
        client = httpx.AsyncClient(timeout=settings.llm.timeout_sec)
        try:
            upstream = await client.send(
                client.build_request("POST", url, json=body), stream=True
            )
        except httpx.TimeoutException:
            await client.aclose()
            lock.release()
            return _error_response(
                504, "Inference backend timed out.", error_type="upstream_timeout"
            )
        except httpx.HTTPError as exc:
            await client.aclose()
            lock.release()
            return _error_response(
                502,
                f"Inference backend is unavailable: {exc.__class__.__name__}",
                error_type="upstream_error",
            )

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
                lock.release()

        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    try:
        async with httpx.AsyncClient(timeout=settings.llm.timeout_sec) as client:
            upstream = await client.post(url, json=body)
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
