"""Silence-aware streaming endpointer for live translation.

Energy-based with an adaptive noise floor: speech segments end after a
configurable silence hangover, are dropped when shorter than a noise-blip
threshold, and are force-cut at a hard maximum so the STT stage never sees
audio past Gemma's 30 s limit. Fully offline — a silero-vad ONNX backend can
replace `_is_speech` later without touching the interface (plan.md B-2/B-6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from localllm.config import TranslateStreamConfig


@dataclass(frozen=True)
class LiveSegment:
    """A completed speech segment, ready for STT."""

    audio: np.ndarray
    start_sample: int
    end_sample: int

    def duration_sec(self, sample_rate: int) -> float:
        return len(self.audio) / sample_rate


class StreamingEndpointer:
    def __init__(
        self,
        sample_rate: int = 16000,
        config: TranslateStreamConfig | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.config = config or TranslateStreamConfig()
        cfg = self.config
        self._frame_len = max(int(sample_rate * cfg.frame_ms / 1000), 1)
        self._hangover_frames = max(int(cfg.hangover_ms / cfg.frame_ms), 1)
        self._pre_roll_frames = max(int(cfg.pre_roll_ms / cfg.frame_ms), 1)
        self._min_segment_frames = max(
            int(cfg.min_segment_seconds * 1000 / cfg.frame_ms), 1
        )
        self._max_segment_frames = max(
            int(cfg.max_segment_seconds * 1000 / cfg.frame_ms), 2
        )

        self._pending = np.zeros(0, dtype=np.float32)
        self._consumed_samples = 0  # absolute position of the next unprocessed frame
        self._noise_floor = cfg.energy_threshold
        self._pre_roll: list[np.ndarray] = []
        self._active: list[np.ndarray] = []
        self._active_start_sample = 0
        self._trailing_silence = 0

    def _is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0
        threshold = max(self.config.energy_threshold, self._noise_floor * 3.0)
        if rms < threshold:
            # Only quiet frames update the noise floor estimate.
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * max(rms, 1e-6)
            return False
        return True

    def feed(self, samples: np.ndarray) -> list[LiveSegment]:
        """Consume incoming PCM (float32 mono); return any completed segments."""
        if samples.size:
            self._pending = np.concatenate(
                [self._pending, np.asarray(samples, dtype=np.float32)]
            )

        segments: list[LiveSegment] = []
        while len(self._pending) >= self._frame_len:
            frame = self._pending[: self._frame_len]
            self._pending = self._pending[self._frame_len :]
            segment = self._process_frame(frame)
            self._consumed_samples += self._frame_len
            if segment is not None:
                segments.append(segment)
        return segments

    def _process_frame(self, frame: np.ndarray) -> LiveSegment | None:
        speech = self._is_speech(frame)

        if not self._active:
            if speech:
                pre_roll = self._pre_roll[-self._pre_roll_frames :]
                self._active = [*pre_roll, frame.copy()]
                self._active_start_sample = self._consumed_samples - len(pre_roll) * self._frame_len
                self._trailing_silence = 0
                self._pre_roll = []
            else:
                self._pre_roll.append(frame.copy())
                if len(self._pre_roll) > self._pre_roll_frames:
                    self._pre_roll.pop(0)
            return None

        self._active.append(frame.copy())
        self._trailing_silence = 0 if speech else self._trailing_silence + 1

        if len(self._active) >= self._max_segment_frames:
            return self._emit(trim_trailing=0)

        if self._trailing_silence >= self._hangover_frames:
            speech_frames = len(self._active) - self._trailing_silence
            if speech_frames >= self._min_segment_frames:
                return self._emit(trim_trailing=max(self._trailing_silence - 2, 0))
            # Too short — noise blip. Discard, keep tail as fresh pre-roll.
            self._pre_roll = self._active[-self._pre_roll_frames :]
            self._active = []
            self._trailing_silence = 0
        return None

    def _emit(self, *, trim_trailing: int) -> LiveSegment:
        frames = self._active[: len(self._active) - trim_trailing] if trim_trailing else self._active
        audio = np.concatenate(frames)
        start = self._active_start_sample
        segment = LiveSegment(
            audio=audio,
            start_sample=start,
            end_sample=start + len(audio),
        )
        # Trimmed trailing silence seeds the next pre-roll window.
        self._pre_roll = self._active[len(frames) :][-self._pre_roll_frames :]
        self._active = []
        self._trailing_silence = 0
        return segment

    def flush(self) -> LiveSegment | None:
        """Stream ended — emit the in-progress segment if it carries speech."""
        if self._pending.size and self._active:
            self._active.append(self._pending.copy())
        self._pending = np.zeros(0, dtype=np.float32)
        if not self._active:
            return None
        audio = np.concatenate(self._active)
        speech_frames = len(self._active) - self._trailing_silence
        min_flush = int(self.config.min_flush_seconds * 1000 / self.config.frame_ms)
        self._active = []
        self._trailing_silence = 0
        if speech_frames < max(min_flush, 1):
            return None
        start = self._active_start_sample
        return LiveSegment(audio=audio, start_sample=start, end_sample=start + len(audio))
