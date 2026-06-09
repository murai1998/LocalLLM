from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from localllm.client.factory import create_llm_client
from localllm.client.protocol import LLMClient
from localllm.config import AppSettings, get_settings
from localllm.media.audio import to_wav_16k

# Whisper split pipeline disabled — Gemma unified audio only.
# from localllm.client.whisper_client import (
#     TranscriptResult,
#     WhisperSTTClient,
#     create_whisper_client,
# )

ToneId = Literal["exact", "professional", "friendly", "cordial"]

LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "pt": "Portuguese",
    "it": "Italian",
    "ko": "Korean",
    "ar": "Arabic",
}

TONE_PRESETS: Final[dict[ToneId, dict[str, str]]] = {
    "exact": {
        "label": "Exact",
        "hint": "Literal and neutral — no stylistic flourish.",
        "instruction": (
            "Use a dry, precise tone. Stay literal and neutral. "
            "Do not add warmth, filler, or embellishment."
        ),
    },
    "professional": {
        "label": "Professional",
        "hint": "Clear, polished language for business settings.",
        "instruction": (
            "Use a professional tone. Be clear, polished, and natural for work contexts."
        ),
    },
    "friendly": {
        "label": "Friendly",
        "hint": "Warm and conversational, still accurate.",
        "instruction": (
            "Use a friendly tone. Sound warm and conversational while staying accurate."
        ),
    },
    "cordial": {
        "label": "Cordial",
        "hint": "Polite, gracious, and personable.",
        "instruction": (
            "Use a cordial tone. Be polite, gracious, and personable without being casual."
        ),
    },
}

DEFAULT_TONE: ToneId = "professional"


@dataclass
class TranslationResult:
    transcript: str
    translation: str
    source_language: str
    target_language: str
    detected_language: str
    llm_elapsed_sec: float
    tone: ToneId = DEFAULT_TONE
    tts_elapsed_sec: float = 0.0

    @property
    def total_elapsed_sec(self) -> float:
        return self.llm_elapsed_sec + self.tts_elapsed_sec


def language_label(code: str | None) -> str:
    if not code:
        return "auto-detected"
    return LANGUAGE_LABELS.get(code, code)


def tone_instruction(tone: ToneId) -> str:
    return TONE_PRESETS.get(tone, TONE_PRESETS[DEFAULT_TONE])["instruction"]


def build_translate_messages(
    transcript: str,
    *,
    source_lang: str | None,
    target_lang: str,
    tone: ToneId = DEFAULT_TONE,
) -> list[dict[str, str]]:
    source_name = language_label(source_lang)
    target_name = language_label(target_lang)
    system = (
        "You are a simultaneous interpreter. "
        f"Translate from {source_name} to {target_name}. "
        f"{tone_instruction(tone)} "
        "Output ONLY the translation with no commentary, labels, or quotes."
    )
    user = f"Source text:\n{transcript.strip()}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_unified_audio_messages(
    audio_path: Path,
    *,
    source_lang: str | None,
    target_lang: str,
    tone: ToneId,
    llm_client: LLMClient,
) -> list[dict[str, Any]]:
    source_name = language_label(source_lang)
    target_name = language_label(target_lang)
    system = (
        "You are a simultaneous interpreter with native audio understanding. "
        f"Listen to the audio, transcribe it in {source_name}, and translate to {target_name}. "
        f"{tone_instruction(tone)} "
        "Respond using exactly this format:\n"
        "TRANSCRIPT:\n"
        "<transcription in source language>\n"
        "TRANSLATION:\n"
        "<translation in target language>"
    )
    user_text = (
        "Transcribe the attached audio and translate it. "
        "Use the TRANSCRIPT / TRANSLATION format exactly."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                llm_client.text_part(user_text),
                llm_client.audio_part(audio_path),
            ],
        },
    ]


def parse_unified_response(text: str) -> tuple[str, str]:
    normalized = text.strip()
    match = re.search(
        r"TRANSCRIPT:\s*(.*?)\s*TRANSLATION:\s*(.*)\Z",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip()

    parts = re.split(r"\n\s*TRANSLATION:\s*\n", normalized, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        transcript = re.sub(r"^TRANSCRIPT:\s*", "", parts[0], flags=re.IGNORECASE).strip()
        return transcript, parts[1].strip()

    raise ValueError(
        "Gemma response did not include TRANSCRIPT / TRANSLATION sections. "
        f"Got: {normalized[:240]}..."
    )


def retranslate_transcript(
    transcript: str,
    *,
    source_lang: str | None = None,
    target_lang: str = "es",
    tone: ToneId = DEFAULT_TONE,
    llm_client: LLMClient | None = None,
    settings: AppSettings | None = None,
) -> TranslationResult:
    """Re-translate existing transcript without re-running speech-to-text."""
    settings = settings or get_settings()
    client = llm_client or create_llm_client(settings)
    source = source_lang if source_lang is not None else settings.translate.source_language
    translation, llm_elapsed = translate_text(
        transcript,
        source_lang=source,
        target_lang=target_lang,
        tone=tone,
        llm_client=client,
        settings=settings,
    )
    detected = source or "auto"
    return TranslationResult(
        transcript=transcript.strip(),
        translation=translation,
        source_language=source or detected,
        target_language=target_lang,
        detected_language=detected,
        llm_elapsed_sec=llm_elapsed,
        tone=tone,
    )


def translate_text(
    transcript: str,
    *,
    source_lang: str | None = None,
    target_lang: str = "es",
    tone: ToneId = DEFAULT_TONE,
    llm_client: LLMClient | None = None,
    settings: AppSettings | None = None,
) -> tuple[str, float]:
    settings = settings or get_settings()
    client = llm_client or create_llm_client(settings)
    started = time.perf_counter()
    translation = client.chat(
        build_translate_messages(
            transcript,
            source_lang=source_lang,
            target_lang=target_lang,
            tone=tone,
        ),
        max_tokens=settings.translate.max_tokens,
        temperature=0.3,
        enable_thinking=False,
    )
    return translation.strip(), time.perf_counter() - started


def translate_audio_unified(
    audio_path: Path,
    *,
    source_lang: str | None = None,
    target_lang: str | None = None,
    tone: ToneId = DEFAULT_TONE,
    llm_client: LLMClient | None = None,
    settings: AppSettings | None = None,
) -> TranslationResult:
    """Transcribe + translate audio in one Gemma multimodal pass."""
    settings = settings or get_settings()
    client = llm_client or create_llm_client(settings)
    target = target_lang or settings.translate.target_language
    source = source_lang if source_lang is not None else settings.translate.source_language

    wav_path = to_wav_16k(audio_path)
    started = time.perf_counter()
    raw = client.chat(
        build_unified_audio_messages(
            wav_path,
            source_lang=source,
            target_lang=target,
            tone=tone,
            llm_client=client,
        ),
        max_tokens=settings.translate.max_tokens,
        temperature=0.3,
        enable_thinking=False,
    )
    llm_elapsed = time.perf_counter() - started
    transcript, translation = parse_unified_response(raw)
    detected = source or "auto"
    return TranslationResult(
        transcript=transcript,
        translation=translation,
        source_language=source or detected,
        target_language=target,
        detected_language=detected,
        llm_elapsed_sec=llm_elapsed,
        tone=tone,
    )


def translate_audio_split(
    audio_path: Path,
    *,
    source_lang: str | None = None,
    target_lang: str | None = None,
    tone: ToneId = DEFAULT_TONE,
    llm_client: LLMClient | None = None,
    settings: AppSettings | None = None,
) -> TranslationResult:
    """Transcribe with ASR_PROMPT + chunking, then translate text (higher quality)."""
    from localllm.pipelines.stt_batch import transcribe_file

    settings = settings or get_settings()
    client = llm_client or create_llm_client(settings)
    target = target_lang or settings.translate.target_language
    source = source_lang if source_lang is not None else settings.translate.source_language

    wav_path = to_wav_16k(audio_path)
    lang_hint = language_label(source) if source else "its original language"

    started = time.perf_counter()
    transcript = transcribe_file(
        wav_path,
        llm_client=client,
        settings=settings,
        language_hint=lang_hint,
    )
    translation, _ = translate_text(
        transcript,
        source_lang=source,
        target_lang=target,
        tone=tone,
        llm_client=client,
        settings=settings,
    )
    llm_elapsed = time.perf_counter() - started
    detected = source or "auto"
    return TranslationResult(
        transcript=transcript.strip(),
        translation=translation,
        source_language=source or detected,
        target_language=target,
        detected_language=detected,
        llm_elapsed_sec=llm_elapsed,
        tone=tone,
    )


def translate_audio(
    audio_path: Path,
    *,
    source_lang: str | None = None,
    target_lang: str | None = None,
    tone: ToneId = DEFAULT_TONE,
    llm_client: LLMClient | None = None,
    settings: AppSettings | None = None,
) -> TranslationResult:
    settings = settings or get_settings()
    pipeline = (settings.translate.pipeline or "split").strip().lower()
    if pipeline == "unified":
        return translate_audio_unified(
            audio_path,
            source_lang=source_lang,
            target_lang=target_lang,
            tone=tone,
            llm_client=llm_client,
            settings=settings,
        )
    return translate_audio_split(
        audio_path,
        source_lang=source_lang,
        target_lang=target_lang,
        tone=tone,
        llm_client=llm_client,
        settings=settings,
    )


# --- Whisper split pipeline (disabled) ---
#
# def translate_audio_split(...):
#     whisper = stt_client or create_whisper_client(settings)
#     transcript_result = whisper.transcribe(audio_path, language=source)
#     ...