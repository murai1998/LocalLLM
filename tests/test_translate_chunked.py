from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from localllm.config import AppSettings, TranslateLiveConfig
from localllm.media.vad import SpeechChunk
from localllm.pipelines.translate_chunked import (
    display_translation_text,
    translate_audio_chunked,
)
from localllm.tts.sentence_queue import SentenceQueue


def _write_speech_wav(path: Path, seconds: float = 3.0) -> None:
    sr = 16000
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr)


def test_translate_audio_chunked(monkeypatch, tmp_path: Path):
    audio = tmp_path / "speech.wav"
    _write_speech_wav(audio, seconds=5.0)

    chunk = SpeechChunk(
        start_sample=0,
        end_sample=16000 * 3,
        audio=np.zeros(16000 * 3, dtype=np.float32),
    )
    monkeypatch.setattr(
        "localllm.pipelines.translate_chunked.to_wav_16k",
        lambda path: path,
    )
    monkeypatch.setattr(
        "localllm.pipelines.translate_chunked.chunk_audio_live",
        lambda *args, **kwargs: [chunk, chunk],
    )
    monkeypatch.setattr(
        "localllm.pipelines.translate_chunked._transcribe_chunk_audio",
        lambda *args, **kwargs: "hello",
    )
    monkeypatch.setattr(
        "localllm.pipelines.translate_chunked.translate_text",
        lambda *args, **kwargs: ("hola", 0.1),
    )

    llm = MagicMock()
    settings = AppSettings()
    settings.translate.live = TranslateLiveConfig()

    result = translate_audio_chunked(
        audio,
        target_lang="es",
        llm_client=llm,
        settings=settings,
    )
    assert result.chunk_count == 2
    assert "hello" in result.transcript
    assert "hola" in result.translation


def test_sentence_queue_new_sentences():
    queue = SentenceQueue()
    queue.spoken_sentence_count = 0
    new = queue.new_sentences("Hello world. How are you?")
    assert new == ["Hello world.", "How are you?"]
    queue.mark_spoken(1)
    newer = queue.new_sentences("Hello world. How are you? I am fine.")
    assert newer == ["How are you?", "I am fine."]


def test_display_translation_text_strips_partial():
    assert display_translation_text("Hola. Como estas sin punto") == "Hola."