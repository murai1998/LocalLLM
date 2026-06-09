from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from localllm.client.factory import create_llm_client
from localllm.client.protocol import LLMClient
from localllm.config import AppSettings, get_settings
from localllm.media.audio import load_mono_16k, merge_transcripts, to_wav_16k, write_wav
from localllm.media.vad import SpeechChunk, _pad_to_min_length, chunk_audio_live
from localllm.pipelines.stt_batch import _transcribe_chunk
from localllm.pipelines.translate import (
    DEFAULT_TONE,
    ToneId,
    language_label,
    translate_text,
)
from localllm.model.prompts import ASR_PROMPT
from localllm.tts.sentence_queue import SentenceQueue, strip_partial_tail

ProgressCallback = Callable[[int, int, "ChunkTranslation"], None]


@dataclass
class ChunkTranslation:
    index: int
    transcript: str
    translation: str
    elapsed_sec: float
    start_sec: float
    end_sec: float


@dataclass
class ChunkedTranslationResult:
    transcript: str
    translation: str
    source_language: str
    target_language: str
    detected_language: str
    llm_elapsed_sec: float
    tone: ToneId = DEFAULT_TONE
    chunks: list[ChunkTranslation] = field(default_factory=list)
    tts_elapsed_sec: float = 0.0

    @property
    def total_elapsed_sec(self) -> float:
        return self.llm_elapsed_sec + self.tts_elapsed_sec

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def _transcribe_chunk_audio(
    chunk: SpeechChunk,
    *,
    sample_rate: int,
    language_hint: str,
    client: LLMClient,
    settings: AppSettings,
    tmp_dir: Path,
    min_chunk_seconds: float,
) -> str:
    audio = _pad_to_min_length(
        np.asarray(chunk.audio, dtype=np.float32),
        sample_rate,
        min_chunk_seconds,
    )
    if audio.size == 0:
        return ""

    wav_path = tmp_dir / f"chunk_{chunk.start_sample}.wav"
    write_wav(wav_path, audio, sample_rate=sample_rate)
    prompt = ASR_PROMPT.replace("its original language", language_hint)
    return _transcribe_chunk(wav_path, prompt=prompt, client=client, settings=settings)


def iter_chunk_translations(
    audio_path: Path,
    *,
    source_lang: str | None = None,
    target_lang: str | None = None,
    tone: ToneId = DEFAULT_TONE,
    llm_client: LLMClient | None = None,
    settings: AppSettings | None = None,
    on_progress: ProgressCallback | None = None,
) -> Iterator[ChunkTranslation]:
    """Yield transcript+translation per VAD chunk (incremental Phase 2)."""
    settings = settings or get_settings()
    client = llm_client or create_llm_client(settings)
    target = target_lang or settings.translate.target_language
    source = source_lang if source_lang is not None else settings.translate.source_language
    lang_hint = language_label(source) if source else "its original language"
    live_cfg = settings.translate.live

    wav_path = to_wav_16k(audio_path)
    cleanup_wav = wav_path.resolve() != Path(audio_path).resolve()
    sr = settings.stt.sample_rate
    try:
        audio = load_mono_16k(wav_path, sample_rate=sr)
        if audio.size == 0:
            return

        chunks = chunk_audio_live(audio, sr, live_cfg)
        if not chunks:
            return

        with tempfile.TemporaryDirectory(prefix="localllm_live_") as tmp:
            tmp_dir = Path(tmp)
            for index, chunk in enumerate(chunks):
                started = time.perf_counter()
                transcript_piece = _transcribe_chunk_audio(
                    chunk,
                    sample_rate=sr,
                    language_hint=lang_hint,
                    client=client,
                    settings=settings,
                    tmp_dir=tmp_dir,
                    min_chunk_seconds=live_cfg.min_chunk_seconds,
                )
                if not transcript_piece.strip():
                    continue

                translation_piece, _ = translate_text(
                    transcript_piece,
                    source_lang=source,
                    target_lang=target,
                    tone=tone,
                    llm_client=client,
                    settings=settings,
                )
                elapsed = time.perf_counter() - started
                item = ChunkTranslation(
                    index=index,
                    transcript=transcript_piece.strip(),
                    translation=translation_piece.strip(),
                    elapsed_sec=elapsed,
                    start_sec=chunk.start_sample / sr,
                    end_sec=chunk.end_sample / sr,
                )
                if on_progress:
                    on_progress(index + 1, len(chunks), item)
                yield item
    finally:
        if cleanup_wav:
            wav_path.unlink(missing_ok=True)


def _fallback_batch_chunks(
    wav_path: Path,
    *,
    source_lang: str | None,
    target_lang: str,
    tone: ToneId,
    client: LLMClient,
    settings: AppSettings,
    lang_hint: str,
) -> list[ChunkTranslation]:
    """Fall back to batch STT windows when per-chunk live STT produced nothing."""
    from localllm.pipelines.stt_batch import transcribe_file

    started = time.perf_counter()
    transcript = transcribe_file(
        wav_path,
        llm_client=client,
        settings=settings,
        language_hint=lang_hint,
    )
    if not transcript.strip():
        return []

    translation, _ = translate_text(
        transcript,
        source_lang=source_lang,
        target_lang=target_lang,
        tone=tone,
        llm_client=client,
        settings=settings,
    )
    elapsed = time.perf_counter() - started
    return [
        ChunkTranslation(
            index=0,
            transcript=transcript.strip(),
            translation=translation.strip(),
            elapsed_sec=elapsed,
            start_sec=0.0,
            end_sec=0.0,
        )
    ]


def translate_audio_chunked(
    audio_path: Path,
    *,
    source_lang: str | None = None,
    target_lang: str | None = None,
    tone: ToneId = DEFAULT_TONE,
    llm_client: LLMClient | None = None,
    settings: AppSettings | None = None,
    on_progress: ProgressCallback | None = None,
) -> ChunkedTranslationResult:
    """VAD chunking → per-chunk ASR → translate → merged partial results."""
    settings = settings or get_settings()
    target = target_lang or settings.translate.target_language
    source = source_lang if source_lang is not None else settings.translate.source_language

    client = llm_client or create_llm_client(settings)
    lang_hint = language_label(source) if source else "its original language"
    started = time.perf_counter()
    chunk_results: list[ChunkTranslation] = []
    transcript_parts: list[str] = []
    translation_parts: list[str] = []

    for item in iter_chunk_translations(
        audio_path,
        source_lang=source,
        target_lang=target,
        tone=tone,
        llm_client=client,
        settings=settings,
        on_progress=on_progress,
    ):
        chunk_results.append(item)
        transcript_parts.append(item.transcript)
        translation_parts.append(item.translation)

    if not chunk_results:
        wav_path = to_wav_16k(audio_path)
        cleanup_wav = wav_path.resolve() != Path(audio_path).resolve()
        try:
            chunk_results = _fallback_batch_chunks(
                wav_path,
                source_lang=source,
                target_lang=target,
                tone=tone,
                client=client,
                settings=settings,
                lang_hint=lang_hint,
            )
            transcript_parts = [item.transcript for item in chunk_results]
            translation_parts = [item.translation for item in chunk_results]
            if chunk_results and on_progress:
                on_progress(1, 1, chunk_results[0])
        finally:
            if cleanup_wav:
                wav_path.unlink(missing_ok=True)

    transcript = merge_transcripts(transcript_parts)
    translation = merge_transcripts(translation_parts)
    detected = source or "auto"

    return ChunkedTranslationResult(
        transcript=transcript,
        translation=translation,
        source_language=source or detected,
        target_language=target,
        detected_language=detected,
        llm_elapsed_sec=time.perf_counter() - started,
        tone=tone,
        chunks=chunk_results,
    )


def synthesize_new_sentences(
    translation: str,
    *,
    target_lang: str,
    voice_id: str | None,
    queue: SentenceQueue,
) -> tuple[bytes | None, list[str], float]:
    """TTS only for newly completed sentences (Phase 2 queue)."""
    from localllm.tts import synthesize_speech

    new_sentences = queue.new_sentences(translation)
    if not new_sentences:
        return queue.pending_audio, [], 0.0

    started = time.perf_counter()
    audio_parts: list[bytes] = []
    for sentence in new_sentences:
        audio_parts.append(
            synthesize_speech(sentence, language=target_lang, voice_id=voice_id)
        )
    combined = audio_parts[0]
    for part in audio_parts[1:]:
        combined = queue.append_wav(combined, part)

    queue.pending_audio = queue.append_wav(queue.pending_audio, combined)
    queue.mark_spoken(len(new_sentences))
    elapsed = time.perf_counter() - started
    return queue.pending_audio, new_sentences, elapsed


def display_translation_text(translation: str) -> str:
    """Hide trailing partial sentence fragment in the UI."""
    return strip_partial_tail(translation) or translation.strip()