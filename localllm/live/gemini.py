"""Remote live-translation engine: Gemini 3.5 Live Translate over the Live API.

A drop-in alternative to ``LiveTranslateSession`` for ``/ws/translate`` and the
live bench, so the local STT → MT → TTS pipeline can be compared against
Google's single-model speech-to-speech translator. It emits the same event
schema (segment / transcript / translation / audio / error / done).

Differences from the local pipeline:
- Gemini segments speech itself, so the raw 16 kHz PCM goes straight up — no
  endpointer. Rows (``index``) advance when Gemini completes a turn or after a
  pause in its output.
- There is no source-language field (auto-detect), and tone / voice options
  don't apply — the model keeps the speaker's own voice characteristics.
- Transcript and translation events carry the row's *cumulative* text, which
  the web UI renders by replacing the row's text.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from localllm.config import AppSettings, get_settings
from localllm.secrets import read_gemini_api_key

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
# (model, config) -> async context manager yielding a Live API session.
ConnectFn = Callable[[str, Any], AbstractAsyncContextManager[Any]]

INPUT_RATE = 16000
OUTPUT_RATE = 24000
# The Live API recommends ~100 ms input chunks.
SEND_CHUNK_BYTES = INPUT_RATE * 2 // 10
# Batch translated audio into ~0.5 s WAVs instead of one event per tiny chunk.
AUDIO_FLUSH_BYTES = OUTPUT_RATE * 2 // 2
# A gap this long in Gemini's output starts a new row.
ROW_GAP_SECONDS = 1.5
# After audio_stream_end, stop waiting once Gemini has been quiet this long.
DRAIN_IDLE_SECONDS = 3.0
DRAIN_MAX_SECONDS = 20.0

# Our ISO-639-1 codes → Gemini BCP-47 codes where they differ.
_GEMINI_LANGUAGE_CODES = {"zh": "zh-Hans"}

# Paid-tier list prices (USD / minute of audio) from ai.google.dev pricing, for
# rough cost readouts in the bench. The free tier is rate-limited but $0.
PRICE_PER_MIN_INPUT = 0.0053
PRICE_PER_MIN_OUTPUT = 0.0315


class GeminiUnavailable(RuntimeError):
    """Raised when the SDK or API key needed for Gemini is missing."""


def gemini_language_code(code: str) -> str:
    return _GEMINI_LANGUAGE_CODES.get(code, code)


def gemini_available() -> bool:
    if read_gemini_api_key() is None:
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _default_connect(api_key: str) -> ConnectFn:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise GeminiUnavailable(
            "google-genai is not installed. Run: pip install -e \".[gemini]\""
        ) from exc
    client = genai.Client(api_key=api_key)

    def connect(model: str, config: Any) -> AbstractAsyncContextManager[Any]:
        return client.aio.live.connect(model=model, config=config)

    return connect


def build_live_config(target_lang: str, *, echo_target_language: bool = False) -> Any:
    from google.genai import types

    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        translation_config=types.TranslationConfig(
            target_language_code=gemini_language_code(target_lang),
            echo_target_language=echo_target_language,
        ),
    )


@dataclass
class _Row:
    index: int
    opened_at: float
    start_sec: float
    transcript: str = ""
    translation: str = ""
    audio: bytearray = field(default_factory=bytearray)
    first_audio_at: float | None = None


class GeminiLiveTranslateSession:
    """One Gemini Live Translate stream: push PCM in, receive events out."""

    def __init__(
        self,
        *,
        target_lang: str,
        on_event: EventCallback,
        settings: AppSettings | None = None,
        api_key: str | None = None,
        connect: ConnectFn | None = None,
        config: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.target_lang = target_lang
        self.on_event = on_event
        self.model = self.settings.gemini.live_translate_model
        if connect is None:
            key = api_key or read_gemini_api_key()
            if not key:
                raise GeminiUnavailable(
                    "No Gemini API key. Set GEMINI_API_KEY (or add it to .env.local) "
                    "and restart the web UI."
                )
            connect = _default_connect(key)
        self._connect = connect
        self._config = config
        self._stack = AsyncExitStack()
        self._session: Any = None
        self._recv_task: asyncio.Task | None = None
        self._pending = bytearray()
        self._sent_bytes = 0
        self._rows: list[_Row] = []
        self._row: _Row | None = None
        self._last_output_at = 0.0
        self._last_message_at = 0.0
        self._output_bytes = 0
        self._input_end = False
        self._closed = False

    # --- lifecycle ---

    async def start(self) -> None:
        config = self._config
        if config is None:
            config = build_live_config(
                self.target_lang,
                echo_target_language=self.settings.gemini.echo_target_language,
            )
        self._session = await self._stack.enter_async_context(self._connect(self.model, config))
        self._last_message_at = time.monotonic()
        self._recv_task = asyncio.create_task(self._receive_loop(), name="gemini-live-recv")

    async def submit_pcm(self, pcm: bytes) -> None:
        """Queue 16 kHz mono Int16 LE PCM; sent upstream in 100 ms chunks."""
        if self._closed:
            return
        self._pending.extend(pcm)
        while len(self._pending) >= SEND_CHUNK_BYTES:
            chunk = bytes(self._pending[:SEND_CHUNK_BYTES])
            del self._pending[:SEND_CHUNK_BYTES]
            await self._send(chunk)

    async def finish(self) -> None:
        """End the input stream, drain remaining translation, emit ``done``."""
        if self._pending:
            await self._send(bytes(self._pending))
            self._pending.clear()
        self._input_end = True
        if self._session is not None and not self._closed:
            with contextlib.suppress(Exception):
                await self._session.send_realtime_input(audio_stream_end=True)
        await self._drain()
        await self._close_row()
        await self._shutdown()
        input_sec = self._sent_bytes / (INPUT_RATE * 2)
        output_sec = self._output_bytes / (OUTPUT_RATE * 2)
        await self.on_event(
            {
                "type": "done",
                "segments": len(self._rows),
                "engine": "gemini",
                "model": self.model,
                "input_audio_sec": round(input_sec, 2),
                "output_audio_sec": round(output_sec, 2),
                "est_cost_usd": round(
                    input_sec / 60 * PRICE_PER_MIN_INPUT + output_sec / 60 * PRICE_PER_MIN_OUTPUT,
                    4,
                ),
            }
        )

    async def abort(self) -> None:
        await self._shutdown()

    # --- internals ---

    async def _send(self, chunk: bytes) -> None:
        if self._session is None or self._closed:
            return
        from google.genai import types

        await self._session.send_realtime_input(
            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={INPUT_RATE}")
        )
        self._sent_bytes += len(chunk)

    async def _drain(self) -> None:
        if self._recv_task is None:
            return
        deadline = time.monotonic() + DRAIN_MAX_SECONDS
        while not self._recv_task.done() and time.monotonic() < deadline:
            if time.monotonic() - self._last_message_at >= DRAIN_IDLE_SECONDS:
                break
            await asyncio.sleep(0.1)

    async def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._recv_task
        with contextlib.suppress(Exception):
            await self._stack.aclose()

    async def _messages(self) -> AsyncIterator[Any]:
        # session.receive() ends after each turn_complete; keep reading until
        # the server closes the stream (an iteration that yields nothing).
        while not self._closed:
            got_any = False
            async for message in self._session.receive():
                got_any = True
                yield message
            if not got_any:
                return

    async def _receive_loop(self) -> None:
        try:
            async for message in self._messages():
                self._last_message_at = time.monotonic()
                await self._handle(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # surface API/session errors to the client
            # The SDK raises APIError(code=1000) even for a normal close.
            if getattr(exc, "code", None) == 1000:
                return
            if not self._closed:
                await self.on_event(
                    {"type": "error", "message": f"Gemini Live: {exc.__class__.__name__}: {exc}"}
                )

    async def _handle(self, message: Any) -> None:
        go_away = getattr(message, "go_away", None)
        if go_away is not None:
            await self.on_event(
                {
                    "type": "error",
                    "message": f"Gemini is closing the session soon (time left: "
                    f"{getattr(go_away, 'time_left', None)}).",
                }
            )
        content = getattr(message, "server_content", None)
        if content is None:
            return
        now = time.monotonic()

        text_in = getattr(getattr(content, "input_transcription", None), "text", None)
        if text_in:
            row = await self._current_row(now)
            row.transcript += text_in
            await self.on_event(
                {"type": "transcript", "index": row.index, "text": row.transcript.strip()}
            )

        text_out = getattr(getattr(content, "output_transcription", None), "text", None)
        if text_out:
            row = await self._current_row(now)
            row.translation += text_out
            self._last_output_at = now
            await self.on_event(
                {"type": "translation", "index": row.index, "text": row.translation.strip()}
            )

        turn = getattr(content, "model_turn", None)
        for part in getattr(turn, "parts", None) or []:
            blob = getattr(part, "inline_data", None)
            data = getattr(blob, "data", None)
            if not data:
                continue
            row = await self._current_row(now)
            if row.first_audio_at is None:
                row.first_audio_at = now
            row.audio.extend(data)
            self._output_bytes += len(data)
            self._last_output_at = now
            if len(row.audio) >= AUDIO_FLUSH_BYTES:
                await self._flush_audio(row)

        if getattr(content, "turn_complete", False) or getattr(
            content, "generation_complete", False
        ):
            await self._close_row()

    async def _current_row(self, now: float) -> _Row:
        row = self._row
        if row is not None and self._last_output_at and now - self._last_output_at > ROW_GAP_SECONDS:
            await self._close_row()
            row = None
        if row is None:
            row = _Row(
                index=len(self._rows),
                opened_at=now,
                start_sec=self._sent_bytes / (INPUT_RATE * 2),
            )
            self._rows.append(row)
            self._row = row
            self._last_output_at = 0.0
            await self.on_event(
                {"type": "segment", "index": row.index, "start_sec": round(row.start_sec, 2)}
            )
        return row

    async def _flush_audio(self, row: _Row) -> None:
        if not row.audio:
            return
        pcm = bytes(row.audio)
        row.audio.clear()
        now = time.monotonic()
        await self.on_event(
            {
                "type": "audio",
                "index": row.index,
                "wav_base64": base64.b64encode(_pcm_to_wav(pcm, OUTPUT_RATE)).decode("ascii"),
                "duration_sec": round(len(pcm) / (OUTPUT_RATE * 2), 2),
                # Gemini streams continuously; "lag" here is time since this row opened.
                "lag_sec": round(now - row.opened_at, 2),
            }
        )

    async def _close_row(self) -> None:
        row = self._row
        if row is None:
            return
        await self._flush_audio(row)
        self._row = None
        await self.on_event(
            {
                "type": "segment",
                "index": row.index,
                "start_sec": round(row.start_sec, 2),
                "end_sec": round(self._sent_bytes / (INPUT_RATE * 2), 2),
            }
        )
