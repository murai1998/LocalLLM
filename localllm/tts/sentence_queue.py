from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass, field

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
PARTIAL_TAIL = re.compile(r"[^.!?…]+$")


@dataclass
class SentenceQueue:
    """Track completed translation sentences for incremental TTS playback."""

    spoken_sentence_count: int = 0
    pending_audio: bytes | None = None
    _history: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.spoken_sentence_count = 0
        self.pending_audio = None
        self._history.clear()

    def split_sentences(self, text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        parts = SENTENCE_END.split(cleaned)
        sentences: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part[-1] not in ".!?…":
                continue
            sentences.append(part)
        return sentences

    def new_sentences(self, translation: str) -> list[str]:
        sentences = self.split_sentences(translation)
        return sentences[self.spoken_sentence_count :]

    def mark_spoken(self, count: int) -> None:
        self.spoken_sentence_count += max(0, count)

    def append_wav(self, existing: bytes | None, new_audio: bytes) -> bytes:
        if not existing:
            return new_audio
        return _concat_wav(existing, new_audio)


def _read_wav_pcm(data: bytes) -> tuple[int, int, int, bytes]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    return sample_rate, channels, sample_width, frames


def _convert_pcm16(
    pcm: bytes,
    *,
    from_rate: int,
    from_channels: int,
    to_rate: int,
    to_channels: int,
) -> bytes:
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    if from_channels > 1:
        # Downmix to mono first; re-expanded to to_channels at the end.
        samples = samples.reshape(-1, from_channels).mean(axis=1)
    samples = samples.astype(np.float64).reshape(-1)

    if from_rate != to_rate and samples.size:
        duration = samples.size / from_rate
        target_len = max(int(round(duration * to_rate)), 1)
        src_t = np.linspace(0.0, duration, samples.size, endpoint=False)
        dst_t = np.linspace(0.0, duration, target_len, endpoint=False)
        samples = np.interp(dst_t, src_t, samples)

    mono = np.clip(samples, -32768, 32767).astype(np.int16)
    if to_channels > 1:
        mono = np.repeat(mono[:, None], to_channels, axis=1).reshape(-1)
    return mono.tobytes()


def _concat_wav(a: bytes, b: bytes) -> bytes:
    sr_a, ch_a, sw_a, pcm_a = _read_wav_pcm(a)
    sr_b, ch_b, sw_b, pcm_b = _read_wav_pcm(b)
    if sw_a != 2 or sw_b != 2:
        raise ValueError("TTS WAV concat supports 16-bit PCM only")
    if sr_a != sr_b or ch_a != ch_b:
        pcm_b = _convert_pcm16(
            pcm_b,
            from_rate=sr_b,
            from_channels=ch_b,
            to_rate=sr_a,
            to_channels=ch_a,
        )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch_a)
        wf.setsampwidth(2)
        wf.setframerate(sr_a)
        wf.writeframes(pcm_a + pcm_b)
    return buf.getvalue()


def strip_partial_tail(text: str) -> str:
    """Drop trailing fragment without sentence-ending punctuation."""
    match = PARTIAL_TAIL.search(text.strip())
    if match and match.group().strip():
        return text[: match.start()].rstrip()
    return text.strip()
