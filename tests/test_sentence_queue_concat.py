import io
import wave

import numpy as np
import pytest

from localllm.tts.sentence_queue import SentenceQueue, _concat_wav, _read_wav_pcm


def _make_wav(seconds: float, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    frames = int(seconds * sample_rate)
    samples = (np.ones(frames * channels) * 1000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        if sampwidth == 2:
            wf.writeframes(samples.tobytes())
        else:
            wf.writeframes(bytes(frames * channels * sampwidth))
    return buf.getvalue()


def _duration(data: bytes) -> float:
    sr, ch, sw, pcm = _read_wav_pcm(data)
    return len(pcm) / (sr * ch * sw)


def test_concat_same_format_appends():
    a = _make_wav(1.0, 22050)
    b = _make_wav(0.5, 22050)
    merged = _concat_wav(a, b)
    assert _duration(merged) == pytest.approx(1.5, abs=0.01)


def test_concat_mismatched_rate_resamples_instead_of_dropping():
    # Mixing a -medium (22050 Hz) and a -low (16000 Hz) voice must not discard
    # the audio accumulated so far.
    a = _make_wav(1.0, 22050)
    b = _make_wav(1.0, 16000)
    merged = _concat_wav(a, b)
    sr, ch, sw, _ = _read_wav_pcm(merged)
    assert sr == 22050
    assert _duration(merged) == pytest.approx(2.0, abs=0.02)


def test_concat_mismatched_channels_downmixes():
    a = _make_wav(1.0, 22050, channels=1)
    b = _make_wav(1.0, 22050, channels=2)
    merged = _concat_wav(a, b)
    sr, ch, sw, _ = _read_wav_pcm(merged)
    assert ch == 1
    assert _duration(merged) == pytest.approx(2.0, abs=0.02)


def test_concat_rejects_non_16bit():
    a = _make_wav(0.2, 22050, sampwidth=2)
    b = _make_wav(0.2, 22050, sampwidth=1)
    with pytest.raises(ValueError):
        _concat_wav(a, b)


def test_queue_append_preserves_existing_audio():
    queue = SentenceQueue()
    a = _make_wav(1.0, 22050)
    b = _make_wav(1.0, 16000)
    merged = queue.append_wav(a, b)
    assert _duration(merged) > 1.5
