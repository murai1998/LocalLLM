"""FastAPI server for the LocalLLM web UI.

Serves the compiled React bundle from ``webui/dist`` and exposes a small REST
API over the existing pipelines. The gateway (``localllm-serve``) must run
separately; every endpoint degrades gracefully when it is offline.
"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from localllm.client.factory import create_llm_client
from localllm.config import ROOT, get_settings
from localllm.devices.resolver import detect_platform
from localllm.pipelines.translate import (
    LANGUAGE_LABELS,
    TONE_PRESETS,
    retranslate_transcript,
)
from localllm.pipelines.translate_chunked import translate_audio_chunked
from localllm.tts import (
    PIPER_AVAILABLE,
    VOICE_OPTIONS,
    synthesize_speech,
    tts_supported,
)

DIST_DIR = ROOT / "webui" / "dist"
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}
OCR_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
MAX_UPLOAD_BYTES = 256 * 1024 * 1024

_llm_client = None
_client_lock = threading.Lock()


def _client():
    global _llm_client
    with _client_lock:
        if _llm_client is None:
            _llm_client = create_llm_client()
        return _llm_client


def _save_upload(upload: UploadFile, allowed: set[str]) -> Path:
    name = Path(upload.filename or "upload")
    suffix = name.suffix.lower()
    if suffix not in allowed:
        raise HTTPException(415, f"Unsupported file type '{suffix or name.name}'.")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds the size limit.")
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="localllm_ui_") as tmp:
        tmp.write(data)
        return Path(tmp.name)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class TranslateTextRequest(BaseModel):
    transcript: str = Field(min_length=1)
    source_lang: str | None = None
    target_lang: str = "es"
    tone: str = "professional"


class TtsRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str
    voice_id: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="LocalLLM Web UI", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "gateway_ready": _client().is_ready(),
            "gateway_url": settings.llm.base_url,
            "model": settings.llm.model,
            "provider": settings.llm.provider,
            "platform": detect_platform(),
            "tts_available": PIPER_AVAILABLE,
            "translate_pipeline": settings.translate.pipeline,
            "live_chunking": {
                "min_chunk_seconds": settings.translate.live.min_chunk_seconds,
                "max_chunk_seconds": settings.translate.live.max_chunk_seconds,
                "overlap_seconds": settings.translate.live.overlap_seconds,
            },
        }

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        settings = get_settings()
        return {
            "languages": LANGUAGE_LABELS,
            "tones": [
                {"id": tone_id, "label": preset["label"], "hint": preset["hint"]}
                for tone_id, preset in TONE_PRESETS.items()
            ],
            "voices": {
                lang: [{"id": o["id"], "label": o["label"]} for o in options]
                for lang, options in VOICE_OPTIONS.items()
            },
            "default_target": settings.translate.target_language,
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are LocalLLM, a helpful local assistant. Answer concisely.",
            }
        ]
        messages += [m.model_dump() for m in req.messages]
        started = time.perf_counter()
        try:
            reply = _client().chat(messages)
        except Exception as exc:
            raise HTTPException(502, f"Inference failed: {exc}") from exc
        return {"reply": reply, "elapsed_sec": time.perf_counter() - started}

    @app.post("/api/translate/audio")
    def translate_audio(
        file: UploadFile,
        source_lang: str = Form(""),
        target_lang: str = Form("es"),
        tone: str = Form("professional"),
    ) -> StreamingResponse:
        """Run chunked translate; stream NDJSON chunk events, then the final result."""
        audio_path = _save_upload(file, AUDIO_SUFFIXES)
        source = source_lang.strip() or None
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def on_progress(done: int, total: int, item) -> None:
            events.put(
                {
                    "type": "chunk",
                    "done": done,
                    "total": total,
                    "index": item.index,
                    "transcript": item.transcript,
                    "translation": item.translation,
                    "elapsed_sec": item.elapsed_sec,
                    "start_sec": item.start_sec,
                    "end_sec": item.end_sec,
                }
            )

        def worker() -> None:
            try:
                result = translate_audio_chunked(
                    audio_path,
                    source_lang=source,
                    target_lang=target_lang,
                    tone=tone,  # type: ignore[arg-type]
                    llm_client=_client(),
                    on_progress=on_progress,
                )
                events.put(
                    {
                        "type": "result",
                        "transcript": result.transcript,
                        "translation": result.translation,
                        "source_language": result.source_language,
                        "target_language": result.target_language,
                        "chunk_count": result.chunk_count,
                        "llm_elapsed_sec": result.llm_elapsed_sec,
                        "tone": result.tone,
                    }
                )
            except Exception as exc:
                events.put({"type": "error", "message": str(exc)})
            finally:
                audio_path.unlink(missing_ok=True)
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def stream():
            while True:
                event = events.get()
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/api/translate/text")
    def translate_text_endpoint(req: TranslateTextRequest) -> dict[str, Any]:
        try:
            result = retranslate_transcript(
                req.transcript,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                tone=req.tone,  # type: ignore[arg-type]
                llm_client=_client(),
            )
        except Exception as exc:
            raise HTTPException(502, f"Translation failed: {exc}") from exc
        return {
            "transcript": result.transcript,
            "translation": result.translation,
            "target_language": result.target_language,
            "llm_elapsed_sec": result.llm_elapsed_sec,
            "tone": result.tone,
        }

    @app.post("/api/tts")
    def tts(req: TtsRequest) -> Response:
        if not PIPER_AVAILABLE:
            raise HTTPException(501, "piper-tts is not installed.")
        if not tts_supported(req.language):
            raise HTTPException(
                422, f"No local Piper voice for language '{req.language}'."
            )
        try:
            audio = synthesize_speech(
                req.text, language=req.language, voice_id=req.voice_id
            )
        except Exception as exc:
            raise HTTPException(500, f"TTS failed: {exc}") from exc
        return Response(content=audio, media_type="audio/wav")

    @app.post("/api/transcribe")
    def transcribe(
        file: UploadFile,
        language_hint: str = Form(""),
    ) -> dict[str, Any]:
        from localllm.pipelines.stt_batch import transcribe_file

        audio_path = _save_upload(file, AUDIO_SUFFIXES)
        started = time.perf_counter()
        try:
            text = transcribe_file(
                audio_path,
                llm_client=_client(),
                language_hint=language_hint.strip() or "its original language",
            )
        except Exception as exc:
            raise HTTPException(502, f"Transcription failed: {exc}") from exc
        finally:
            audio_path.unlink(missing_ok=True)
        return {"text": text, "elapsed_sec": time.perf_counter() - started}

    @app.post("/api/ocr")
    def ocr(
        file: UploadFile,
        instructions: str = Form(""),
    ) -> dict[str, Any]:
        from localllm.pipelines.ocr import process_path

        doc_path = _save_upload(file, OCR_SUFFIXES)
        started = time.perf_counter()
        try:
            result = process_path(doc_path, instructions=instructions)
        except Exception as exc:
            raise HTTPException(502, f"OCR failed: {exc}") from exc
        finally:
            doc_path.unlink(missing_ok=True)
        result["elapsed_sec"] = time.perf_counter() - started
        return result

    # --- Static SPA bundle ---
    if DIST_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = (DIST_DIR / path).resolve()
            if (
                path
                and candidate.is_file()
                and candidate.is_relative_to(DIST_DIR.resolve())
            ):
                return FileResponse(candidate)
            return FileResponse(DIST_DIR / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def missing_bundle() -> Response:
            return Response(
                "Web UI bundle not found. Build it with:\n\n"
                "  cd webui && npm install && npm run build\n",
                media_type="text/plain",
                status_code=503,
            )

    return app


app = create_app()
