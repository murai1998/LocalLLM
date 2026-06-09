from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from localllm.config import AppSettings, TranslateLiveConfig
from localllm.pipelines.translate_chunked import translate_audio_chunked


def test_translate_audio_chunked_falls_back_to_batch_stt(tmp_path: Path, monkeypatch):
    audio = tmp_path / "speech.wav"
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3), endpoint=False)
    sf.write(audio, 0.2 * np.sin(2 * np.pi * 440 * t), sr)

    monkeypatch.setattr(
        "localllm.pipelines.translate_chunked.to_wav_16k",
        lambda path: path,
    )
    monkeypatch.setattr(
        "localllm.pipelines.translate_chunked.chunk_audio_live",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "localllm.pipelines.translate_chunked._transcribe_chunk_audio",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(
        "localllm.pipelines.stt_batch.transcribe_file",
        lambda *args, **kwargs: "hello from batch",
    )
    monkeypatch.setattr(
        "localllm.pipelines.translate_chunked.translate_text",
        lambda *args, **kwargs: ("hola batch", 0.1),
    )

    settings = AppSettings()
    settings.translate.live = TranslateLiveConfig()

    result = translate_audio_chunked(
        audio,
        target_lang="es",
        llm_client=MagicMock(),
        settings=settings,
    )
    assert result.chunk_count == 1
    assert result.transcript == "hello from batch"
    assert result.translation == "hola batch"