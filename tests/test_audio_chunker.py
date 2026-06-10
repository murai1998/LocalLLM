import numpy as np

from localllm.config import SttConfig
from localllm.media.audio import chunk_audio, merge_transcripts


def test_chunk_audio_respects_max_duration():
    sr = 16000
    audio = np.zeros(sr * 90, dtype=np.float32)
    cfg = SttConfig(chunk_seconds=28, overlap_seconds=2, max_chunk_seconds=30)
    chunks = chunk_audio(audio, sr, cfg)
    assert len(chunks) >= 3
    assert all(len(c) <= sr * 30 for c in chunks)


def test_merge_transcripts_overlap():
    a = "hello world this is a test"
    b = "this is a test of merging"
    merged = merge_transcripts([a, b])
    assert "hello world" in merged
    assert "merging" in merged
    assert merged.count("this is a test") == 1
