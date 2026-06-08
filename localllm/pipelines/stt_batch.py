from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from localllm.client.factory import create_llm_client
from localllm.client.protocol import LLMClient
from localllm.config import AppSettings, get_settings
from localllm.media.audio import chunk_audio, load_mono_16k, merge_transcripts, write_wav
from localllm.model.prompts import ASR_PROMPT

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}


def _transcribe_chunk(
    wav_path: Path,
    *,
    prompt: str,
    client: LLMClient,
    settings: AppSettings,
) -> str:
    """Stateless single-chunk transcription (no conversation history pollution)."""
    gen = settings.generation
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You transcribe speech accurately."},
        {
            "role": "user",
            "content": [
                client.text_part(prompt),
                client.audio_part(wav_path),
            ],
        },
    ]
    return client.chat(
        messages,
        max_tokens=gen.stt_max_tokens,
        temperature=gen.stt_temperature,
        enable_thinking=False,
    ).strip()


def transcribe_file(
    path: Path,
    *,
    llm_client: LLMClient | None = None,
    settings: AppSettings | None = None,
    language_hint: str = "its original language",
) -> str:
    settings = settings or get_settings()
    client = llm_client or create_llm_client(settings)

    sr = settings.stt.sample_rate
    audio = load_mono_16k(path, sample_rate=sr)
    chunks = chunk_audio(audio, sr, settings.stt)

    prompt = ASR_PROMPT.replace("its original language", language_hint)
    parts: list[str] = []

    with tempfile.TemporaryDirectory(prefix="localllm_stt_") as tmp:
        tmp_path = Path(tmp)
        for i, chunk in enumerate(chunks):
            wav = tmp_path / f"chunk_{i:03d}.wav"
            write_wav(wav, chunk, sample_rate=sr)
            text = _transcribe_chunk(wav, prompt=prompt, client=client, settings=settings)
            if text:
                parts.append(text)

    return merge_transcripts(parts)


def transcribe_to_file(
    input_path: Path,
    output_path: Path | None = None,
    *,
    llm_client: LLMClient | None = None,
    settings: AppSettings | None = None,
) -> Path:
    if input_path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported audio file: {input_path}")
    text = transcribe_file(input_path, llm_client=llm_client, settings=settings)
    out = output_path or input_path.with_suffix(".txt")
    out.write_text(text + "\n", encoding="utf-8")
    return out