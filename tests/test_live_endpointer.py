import numpy as np

from localllm.config import TranslateStreamConfig
from localllm.live import StreamingEndpointer

SR = 16000


def _tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def _cfg(**overrides) -> TranslateStreamConfig:
    base = {
        "hangover_ms": 600,
        "min_segment_seconds": 1.0,
        "max_segment_seconds": 8.0,
        "pre_roll_ms": 300,
    }
    base.update(overrides)
    return TranslateStreamConfig(**base)


def _feed_in_chunks(ep: StreamingEndpointer, audio: np.ndarray, chunk_ms: int = 250):
    segments = []
    chunk = int(SR * chunk_ms / 1000)
    for off in range(0, len(audio), chunk):
        segments.extend(ep.feed(audio[off : off + chunk]))
    return segments


def test_speech_then_silence_emits_one_segment():
    ep = StreamingEndpointer(SR, _cfg())
    audio = np.concatenate([_silence(0.5), _tone(3.0), _silence(1.5)])
    segments = _feed_in_chunks(ep, audio)
    assert len(segments) == 1
    seg = segments[0]
    # Segment covers the speech plus pre-roll, not the whole stream.
    assert 2.8 <= seg.duration_sec(SR) <= 4.2
    assert seg.start_sample <= int(0.5 * SR)


def test_two_utterances_split_on_silence():
    ep = StreamingEndpointer(SR, _cfg())
    audio = np.concatenate([_tone(2.0), _silence(1.2), _tone(2.0), _silence(1.2)])
    segments = _feed_in_chunks(ep, audio)
    assert len(segments) == 2


def test_continuous_speech_is_force_cut_at_max():
    ep = StreamingEndpointer(SR, _cfg(max_segment_seconds=4.0))
    segments = _feed_in_chunks(ep, _tone(10.0))
    tail = ep.flush()
    if tail is not None:
        segments.append(tail)
    assert len(segments) >= 2
    assert all(s.duration_sec(SR) <= 4.1 for s in segments)


def test_noise_blip_is_dropped():
    ep = StreamingEndpointer(SR, _cfg(min_segment_seconds=1.5))
    audio = np.concatenate([_tone(0.3), _silence(2.0)])
    assert _feed_in_chunks(ep, audio) == []
    assert ep.flush() is None


def test_silence_only_emits_nothing():
    ep = StreamingEndpointer(SR, _cfg())
    assert _feed_in_chunks(ep, _silence(5.0)) == []
    assert ep.flush() is None


def test_flush_emits_trailing_speech():
    ep = StreamingEndpointer(SR, _cfg())
    segments = _feed_in_chunks(ep, _tone(2.0))  # no trailing silence yet
    assert segments == []
    tail = ep.flush()
    assert tail is not None
    assert tail.duration_sec(SR) >= 1.5


def test_quiet_speech_detected_despite_loud_transient():
    ep = StreamingEndpointer(SR, _cfg())
    audio = np.concatenate(
        [_silence(0.5), _tone(0.2, amp=0.9), _silence(1.0), _tone(2.0, amp=0.05), _silence(1.0)]
    )
    segments = _feed_in_chunks(ep, audio)
    # The loud 0.2s clap is dropped as a blip; the quiet 2s speech is kept.
    assert len(segments) == 1
    assert segments[0].duration_sec(SR) >= 1.8
