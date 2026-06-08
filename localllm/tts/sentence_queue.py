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


def _read_wav_pcm(data: bytes) -> tuple[int, int, bytes]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return sample_rate, channels, frames


def _concat_wav(a: bytes, b: bytes) -> bytes:
    sr_a, ch_a, pcm_a = _read_wav_pcm(a)
    sr_b, ch_b, pcm_b = _read_wav_pcm(b)
    if sr_a != sr_b or ch_a != ch_b:
        return b
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