"""Pipelined STT → MT → TTS session for streaming voice-to-voice translation.

Three asyncio workers connected by queues: while segment N is being
translated/spoken, segment N+1 is already in STT (the gateway allows 2
concurrent inference requests). Stage functions are injectable so the bench
harness and tests can run without a GPU.
"""

from __future__ import annotations

import asyncio
import base64
import io
import tempfile
import time
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from localllm.client.protocol import LLMClient
from localllm.config import AppSettings, get_settings
from localllm.media.audio import write_wav
from localllm.media.vad import _pad_to_min_length
from localllm.model.prompts import ASR_PROMPT
from localllm.pipelines.translate import build_translate_messages, language_label

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]

_QUEUE_SIZE = 8


@dataclass
class SegmentJob:
    index: int
    audio: np.ndarray
    sample_rate: int
    start_sec: float
    end_sec: float
    captured_at: float  # time.monotonic() when the segment was completed


def _wav_duration_sec(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or 1)


class LiveTranslateSession:
    """One live translation stream: feed segments in, await events out."""

    def __init__(
        self,
        *,
        source_lang: str | None = None,
        target_lang: str = "es",
        tone: str = "professional",
        voice_id: str | None = None,
        on_event: EventCallback,
        settings: AppSettings | None = None,
        llm_client: LLMClient | None = None,
        stt_fn: Callable[[SegmentJob], str] | None = None,
        mt_fn: Callable[[str, list[tuple[str, str]]], str] | None = None,
        tts_fn: Callable[[str], bytes | None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = llm_client
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tone = tone
        self.voice_id = voice_id
        self.on_event = on_event

        self._stt = stt_fn or self._default_stt
        self._mt = mt_fn or self._default_mt
        self._tts = tts_fn or self._default_tts

        self._stt_q: asyncio.Queue[SegmentJob | None] = asyncio.Queue(maxsize=_QUEUE_SIZE)
        self._mt_q: asyncio.Queue[tuple[SegmentJob, str] | None] = asyncio.Queue(
            maxsize=_QUEUE_SIZE
        )
        self._tts_q: asyncio.Queue[tuple[SegmentJob, str] | None] = asyncio.Queue(
            maxsize=_QUEUE_SIZE
        )
        self._context: deque[tuple[str, str]] = deque(maxlen=2)
        self._tasks: list[asyncio.Task] = []
        self._index = 0

    # --- lifecycle ---

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._stt_worker(), name="live-stt"),
            asyncio.create_task(self._mt_worker(), name="live-mt"),
            asyncio.create_task(self._tts_worker(), name="live-tts"),
        ]

    async def submit(self, audio: np.ndarray, *, sample_rate: int, start_sample: int) -> None:
        job = SegmentJob(
            index=self._index,
            audio=audio,
            sample_rate=sample_rate,
            start_sec=start_sample / sample_rate,
            end_sec=(start_sample + len(audio)) / sample_rate,
            captured_at=time.monotonic(),
        )
        self._index += 1
        await self.on_event(
            {
                "type": "segment",
                "index": job.index,
                "start_sec": round(job.start_sec, 2),
                "end_sec": round(job.end_sec, 2),
                "duration_sec": round(len(audio) / sample_rate, 2),
            }
        )
        await self._stt_q.put(job)

    async def finish(self) -> None:
        """Drain the pipeline and emit a final `done` event."""
        await self._stt_q.put(None)
        await asyncio.gather(*self._tasks)
        await self.on_event({"type": "done", "segments": self._index})

    async def abort(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # --- workers ---

    async def _stt_worker(self) -> None:
        while True:
            job = await self._stt_q.get()
            if job is None:
                await self._mt_q.put(None)
                return
            started = time.perf_counter()
            try:
                text = await asyncio.to_thread(self._stt, job)
            except Exception as exc:
                await self._emit_error(job.index, "stt", exc)
                continue
            await self.on_event(
                {
                    "type": "transcript",
                    "index": job.index,
                    "text": text,
                    "elapsed_sec": round(time.perf_counter() - started, 2),
                }
            )
            if text.strip():
                await self._mt_q.put((job, text))

    async def _mt_worker(self) -> None:
        while True:
            item = await self._mt_q.get()
            if item is None:
                await self._tts_q.put(None)
                return
            job, transcript = item
            started = time.perf_counter()
            try:
                translation = await asyncio.to_thread(
                    self._mt, transcript, list(self._context)
                )
            except Exception as exc:
                await self._emit_error(job.index, "translate", exc)
                continue
            self._context.append((transcript, translation))
            await self.on_event(
                {
                    "type": "translation",
                    "index": job.index,
                    "text": translation,
                    "elapsed_sec": round(time.perf_counter() - started, 2),
                }
            )
            if translation.strip():
                await self._tts_q.put((job, translation))

    async def _tts_worker(self) -> None:
        while True:
            item = await self._tts_q.get()
            if item is None:
                return
            job, translation = item
            started = time.perf_counter()
            try:
                wav_bytes = await asyncio.to_thread(self._tts, translation)
            except Exception as exc:
                await self._emit_error(job.index, "tts", exc)
                continue
            if wav_bytes is None:
                continue
            await self.on_event(
                {
                    "type": "audio",
                    "index": job.index,
                    "wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
                    "duration_sec": round(_wav_duration_sec(wav_bytes), 2),
                    "elapsed_sec": round(time.perf_counter() - started, 2),
                    "lag_sec": round(time.monotonic() - job.captured_at, 2),
                }
            )

    async def _emit_error(self, index: int, stage: str, exc: Exception) -> None:
        await self.on_event(
            {"type": "error", "index": index, "stage": stage, "message": str(exc)}
        )

    # --- default stage implementations (real models) ---

    def _llm(self) -> LLMClient:
        if self._client is None:
            from localllm.client.factory import create_llm_client

            self._client = create_llm_client(self.settings)
        return self._client

    def _default_stt(self, job: SegmentJob) -> str:
        from localllm.pipelines.stt_batch import _transcribe_chunk

        audio = _pad_to_min_length(
            job.audio, job.sample_rate, self.settings.translate.stream.stt_pad_seconds
        )
        lang_hint = (
            language_label(self.source_lang) if self.source_lang else "its original language"
        )
        prompt = ASR_PROMPT.replace("its original language", lang_hint)
        with tempfile.TemporaryDirectory(prefix="localllm_live_") as tmp:
            wav_path = Path(tmp) / f"segment_{job.index}.wav"
            write_wav(wav_path, audio, sample_rate=job.sample_rate)
            return _transcribe_chunk(
                wav_path, prompt=prompt, client=self._llm(), settings=self.settings
            )

    def _default_mt(self, transcript: str, context: list[tuple[str, str]]) -> str:
        messages = build_translate_messages(
            transcript,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            tone=self.tone,  # type: ignore[arg-type]
        )
        if context:
            ctx_lines = "\n".join(f"- {src} → {tgt}" for src, tgt in context)
            messages[-1]["content"] = (
                "Recent segments already translated (context only — do not repeat):\n"
                f"{ctx_lines}\n\n{messages[-1]['content']}"
            )
        return self._llm().chat(
            messages,
            max_tokens=self.settings.translate.max_tokens,
            temperature=0.3,
            enable_thinking=False,
        ).strip()

    def _default_tts(self, translation: str) -> bytes | None:
        from localllm.tts import PIPER_AVAILABLE, synthesize_speech, tts_supported

        if not PIPER_AVAILABLE or not tts_supported(self.target_lang):
            return None
        return synthesize_speech(
            translation, language=self.target_lang, voice_id=self.voice_id
        )
