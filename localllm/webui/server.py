"""FastAPI server for the LocalLLM web UI.

Serves the compiled React bundle from ``webui/dist`` and exposes a small REST
API over the existing pipelines. The gateway (``localllm-serve``) must run
separately; every endpoint degrades gracefully when it is offline.
"""

from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from localllm.client.factory import create_llm_client
from localllm.config import ROOT, get_settings
from localllm.devices.resolver import detect_platform
from localllm.live import LiveTranslateSession, StreamingEndpointer
from localllm.media.attachments import (
    AGENT_ALLOWED,
    CHAT_ALLOWED,
    AttachmentError,
    attachment_kind,
    build_multimodal_content,
    prepare_agent_context,
    prepare_chat_turn,
)
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
ATTACHMENT_SUFFIXES = CHAT_ALLOWED | AGENT_ALLOWED
MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_ACTIVE_UPLOADS = 16
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "localllm" / "webui_uploads"
UPLOAD_MAX_AGE_SECONDS = 24 * 3600

_llm_client = None
_client_lock = threading.Lock()


def _client():
    global _llm_client
    with _client_lock:
        if _llm_client is None:
            _llm_client = create_llm_client()
        return _llm_client


@dataclass
class StoredUpload:
    id: str
    name: str
    kind: str
    size: int
    path: Path


_uploads: dict[str, StoredUpload] = {}
_uploads_lock = threading.Lock()


def sweep_stale_uploads(
    root: Path = UPLOAD_ROOT, max_age_seconds: float = UPLOAD_MAX_AGE_SECONDS
) -> int:
    """Remove attachment dirs left behind by previous server runs."""
    if not root.is_dir():
        return 0
    now = time.time()
    removed = 0
    for entry in root.iterdir():
        try:
            if entry.is_dir() and now - entry.stat().st_mtime > max_age_seconds:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def _store_attachment(upload: UploadFile) -> StoredUpload:
    name = Path(upload.filename or "file").name
    suffix = Path(name).suffix.lower()
    if suffix not in ATTACHMENT_SUFFIXES:
        raise HTTPException(415, f"Unsupported attachment type '{suffix or name}'.")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Attachment exceeds the size limit.")
    if not data:
        raise HTTPException(400, f"Attachment '{name}' is empty.")

    upload_id = uuid.uuid4().hex[:12]
    dest_dir = UPLOAD_ROOT / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    dest.write_bytes(data)
    return StoredUpload(
        id=upload_id, name=name, kind=attachment_kind(suffix), size=len(data), path=dest
    )


def _attachment_paths(ids: list[str]) -> list[Path]:
    paths: list[Path] = []
    with _uploads_lock:
        for attachment_id in ids:
            stored = _uploads.get(attachment_id)
            if stored is None or not stored.path.is_file():
                raise HTTPException(
                    404, f"Attachment '{attachment_id}' not found — re-attach the file."
                )
            paths.append(stored.path)
    return paths


# Agent graphs are cached per enabled-skill set (compilation is cheap; the
# ChatEngine inside holds no conversation state between requests).
_agent_graphs: dict[tuple[str, ...], Any] = {}
_agent_lock = threading.Lock()


def _agent_graph(skill_names: tuple[str, ...]):
    from localllm.agents import build_agent_graph
    from localllm.agents.skills import resolve_skills

    with _agent_lock:
        graph = _agent_graphs.get(skill_names)
        if graph is None:
            graph = build_agent_graph(
                autostart_server=False,
                skills=resolve_skills(list(skill_names)),
            )
            _agent_graphs[skill_names] = graph
        return graph


def _agent_trace(messages: list) -> list[dict[str, str]]:
    from langchain_core.messages import AIMessage, ToolMessage

    trace: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                trace.append(
                    {
                        "kind": "call",
                        "title": tc["name"],
                        "body": json.dumps(tc.get("args") or {}, ensure_ascii=False),
                    }
                )
        elif isinstance(msg, ToolMessage):
            body = str(msg.content)
            trace.append(
                {
                    "kind": "result",
                    "title": str(msg.name or "tool"),
                    "body": body[:2000] + ("…" if len(body) > 2000 else ""),
                }
            )
    return trace


def _agent_final_reply(messages: list) -> str:
    from langchain_core.messages import AIMessage

    for m in reversed(messages):
        if not isinstance(m, AIMessage) or not m.content:
            continue
        if getattr(m, "tool_calls", None):
            continue
        if "<|tool_call>" in str(m.content):
            continue
        return str(m.content)
    return "(The agent did not produce a final answer — try rephrasing.)"


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
    mode: Literal["chat", "agent"] = "chat"
    skills: list[str] = Field(default_factory=list)
    attachment_ids: list[str] = Field(default_factory=list)


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
    sweep_stale_uploads()

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
            "live_stream": {
                "hangover_ms": settings.translate.stream.hangover_ms,
                "min_segment_seconds": settings.translate.stream.min_segment_seconds,
                "max_segment_seconds": settings.translate.stream.max_segment_seconds,
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

    @app.get("/api/skills")
    def skills() -> list[dict[str, str]]:
        from localllm.agents import discover_skills

        return [
            {"name": skill.name, "description": skill.description}
            for skill in discover_skills()
        ]

    @app.post("/api/uploads")
    def upload_attachments(files: list[UploadFile]) -> list[dict[str, Any]]:
        with _uploads_lock:
            if len(_uploads) + len(files) > MAX_ACTIVE_UPLOADS:
                raise HTTPException(
                    409,
                    f"Too many active attachments (max {MAX_ACTIVE_UPLOADS}) — detach some first.",
                )
        stored_items = [_store_attachment(f) for f in files]
        with _uploads_lock:
            for item in stored_items:
                _uploads[item.id] = item
        return [
            {"id": s.id, "name": s.name, "kind": s.kind, "size": s.size}
            for s in stored_items
        ]

    @app.delete("/api/uploads/{attachment_id}")
    def delete_attachment(attachment_id: str) -> dict[str, bool]:
        with _uploads_lock:
            stored = _uploads.pop(attachment_id, None)
        if stored is None:
            raise HTTPException(404, "Attachment not found.")
        shutil.rmtree(stored.path.parent, ignore_errors=True)
        return {"deleted": True}

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        if req.messages[-1].role != "user":
            raise HTTPException(400, "The last message must be from the user.")
        attachment_paths = _attachment_paths(req.attachment_ids)
        last_text = req.messages[-1].content
        prior = [m.model_dump() for m in req.messages[:-1]]
        started = time.perf_counter()

        if req.mode == "agent":
            return _run_agent(req, last_text, prior, attachment_paths, started)

        try:
            user_text, image_paths, audio_path = prepare_chat_turn(
                last_text, attachment_paths
            )
        except AttachmentError as exc:
            raise HTTPException(422, str(exc)) from exc
        content = build_multimodal_content(
            user_text,
            image_paths=image_paths,
            audio_path=audio_path,
            client=_client(),
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are LocalLLM, a helpful local assistant. Answer concisely.",
            },
            *prior,
            {"role": "user", "content": content},
        ]
        try:
            reply = _client().chat(messages)
        except Exception as exc:
            raise HTTPException(502, f"Inference failed: {exc}") from exc
        return {
            "reply": reply,
            "elapsed_sec": time.perf_counter() - started,
            "mode": "chat",
            "steps": [],
        }

    def _run_agent(
        req: ChatRequest,
        last_text: str,
        prior: list[dict[str, Any]],
        attachment_paths: list[Path],
        started: float,
    ) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, HumanMessage

        from localllm.agents import invoke_agent

        try:
            graph = _agent_graph(tuple(sorted(set(req.skills))))
        except ValueError as exc:  # unknown skill name
            raise HTTPException(422, str(exc)) from exc

        try:
            user_text, image_paths, audio_path = prepare_agent_context(
                last_text, attachment_paths
            )
        except AttachmentError as exc:
            raise HTTPException(422, str(exc)) from exc
        content = build_multimodal_content(
            user_text,
            image_paths=image_paths,
            audio_path=audio_path,
            client=_client(),
        )

        lc_messages = [
            HumanMessage(content=m["content"])
            if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in prior
        ]
        lc_messages.append(HumanMessage(content=content))

        try:
            result = invoke_agent(graph, {"messages": lc_messages})
        except Exception as exc:
            raise HTTPException(502, f"Agent run failed: {exc}") from exc

        return {
            "reply": _agent_final_reply(result["messages"]),
            "elapsed_sec": time.perf_counter() - started,
            "mode": "agent",
            "steps": _agent_trace(result["messages"]),
        }

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

    @app.websocket("/ws/translate")
    async def ws_translate(ws: WebSocket) -> None:
        """Streaming voice-to-voice translation (plan.md Workstream B).

        Protocol — client sends a JSON `start` message, then binary frames of
        16 kHz mono **Int16 LE** PCM, then a JSON `stop`. Server emits JSON
        events: segment / transcript / translation / audio (base64 WAV) /
        error / done.
        """
        await ws.accept()
        settings = get_settings()
        try:
            start = await ws.receive_json()
        except (WebSocketDisconnect, ValueError):
            await ws.close(code=1003)
            return
        if start.get("type") != "start":
            await ws.send_json({"type": "error", "message": "First message must be 'start'."})
            await ws.close(code=1003)
            return

        sample_rate = int(start.get("sample_rate") or 16000)
        source_lang = (start.get("source_lang") or "").strip() or None
        target_lang = (start.get("target_lang") or settings.translate.target_language).strip()
        tone = (start.get("tone") or "professional").strip()
        voice_id = (start.get("voice_id") or "").strip() or None

        endpointer = StreamingEndpointer(sample_rate, settings.translate.stream)
        session = LiveTranslateSession(
            source_lang=source_lang,
            target_lang=target_lang,
            tone=tone,
            voice_id=voice_id,
            settings=settings,
            llm_client=_client(),
            on_event=ws.send_json,
        )
        await session.start()

        async def submit(segment) -> None:
            await session.submit(
                segment.audio, sample_rate=sample_rate, start_sample=segment.start_sample
            )

        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code") or 1000)
                if message.get("bytes") is not None:
                    pcm = np.frombuffer(message["bytes"], dtype="<i2")
                    samples = pcm.astype(np.float32) / 32768.0
                    for segment in endpointer.feed(samples):
                        await submit(segment)
                elif message.get("text") is not None:
                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") == "stop":
                        tail = endpointer.flush()
                        if tail is not None:
                            await submit(tail)
                        await session.finish()
                        break
        except WebSocketDisconnect:
            await session.abort()
            return
        await ws.close()

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
