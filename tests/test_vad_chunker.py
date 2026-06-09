import numpy as np

from localllm.config import TranslateLiveConfig
from localllm.media.vad import chunk_audio_live, chunk_audio_vad


def _tone(sr: int, seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_vad_chunks_speech_audio():
    sr = 16000
    speech = _tone(sr, 3.0)
    silence = np.zeros(int(sr * 1.0), dtype=np.float32)
    audio = np.concatenate([speech, silence, speech, silence, speech])
    cfg = TranslateLiveConfig(min_chunk_seconds=2.0, max_chunk_seconds=4.0, overlap_seconds=0.5)
    chunks = chunk_audio_vad(audio, sr, cfg)
    assert len(chunks) >= 1
    assert all(2.0 * sr * 0.4 <= len(c.audio) <= 4.0 * sr + 1 for c in chunks)


def test_live_chunks_use_nine_to_ten_second_windows():
    sr = 16000
    audio = _tone(sr, 25.0)
    cfg = TranslateLiveConfig(min_chunk_seconds=9.0, max_chunk_seconds=10.0, overlap_seconds=1.0)
    chunks = chunk_audio_live(audio, sr, cfg)
    assert len(chunks) >= 2
    assert all(len(c.audio) >= 9.0 * sr for c in chunks)


def test_live_chunks_pad_short_final_clip():
    sr = 16000
    audio = _tone(sr, 5.0)
    cfg = TranslateLiveConfig(min_chunk_seconds=9.0, max_chunk_seconds=10.0, overlap_seconds=1.0)
    chunks = chunk_audio_live(audio, sr, cfg)
    assert len(chunks) == 1
    assert len(chunks[0].audio) == 9 * sr


def test_vad_fixed_windows_for_silence():
    sr = 16000
    audio = np.zeros(sr * 6, dtype=np.float32)
    cfg = TranslateLiveConfig(min_chunk_seconds=2.0, max_chunk_seconds=4.0, overlap_seconds=0.5)
    chunks = chunk_audio_vad(audio, sr, cfg)
    assert len(chunks) >= 1