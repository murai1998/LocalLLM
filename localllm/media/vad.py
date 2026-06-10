from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from localllm.config import TranslateLiveConfig


@dataclass(frozen=True)
class SpeechChunk:
    """A slice of audio samples ready for Gemma STT."""

    start_sample: int
    end_sample: int
    audio: np.ndarray


def _frame_energies(audio: np.ndarray, sample_rate: int, frame_ms: int) -> np.ndarray:
    frame_len = max(int(sample_rate * frame_ms / 1000), 1)
    if len(audio) < frame_len:
        return np.array([float(np.sqrt(np.mean(audio**2)))] if len(audio) else [0.0])
    trims = len(audio) - (len(audio) % frame_len)
    frames = audio[:trims].reshape(-1, frame_len)
    return np.sqrt(np.mean(frames**2, axis=1))


def _speech_mask(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int,
    energy_threshold: float,
) -> np.ndarray:
    energies = _frame_energies(audio, sample_rate, frame_ms)
    if energies.size == 0:
        return np.zeros(0, dtype=bool)
    # Adaptive threshold anchored to the noise floor (20th percentile) rather than
    # the peak alone, so one loud transient cannot suppress quiet speech, and
    # silence-only audio (noise floor ≈ peak) stays below the absolute threshold.
    peak = float(np.max(energies)) or 1.0
    floor = float(np.percentile(energies, 20))
    threshold = max(energy_threshold, min(peak * 0.05, floor * 3.0))
    return energies >= threshold


def _mask_to_regions(mask: np.ndarray, frame_len: int, total_samples: int) -> list[tuple[int, int]]:
    if mask.size == 0:
        return [(0, total_samples)] if total_samples else []

    regions: list[tuple[int, int]] = []
    in_speech = False
    start_frame = 0
    for idx, active in enumerate(mask):
        if active and not in_speech:
            in_speech = True
            start_frame = idx
        elif not active and in_speech:
            regions.append((start_frame * frame_len, min(idx * frame_len, total_samples)))
            in_speech = False
    if in_speech:
        regions.append((start_frame * frame_len, total_samples))
    return regions


def _fixed_windows(
    audio: np.ndarray,
    sample_rate: int,
    config: TranslateLiveConfig,
) -> list[SpeechChunk]:
    chunk_len = int(config.max_chunk_seconds * sample_rate)
    overlap = int(config.overlap_seconds * sample_rate)
    step = max(chunk_len - overlap, 1)
    chunks: list[SpeechChunk] = []
    for start in range(0, len(audio), step):
        end = min(start + chunk_len, len(audio))
        piece = audio[start:end]
        if len(piece) < int(sample_rate * config.min_chunk_seconds * 0.5):
            break
        chunks.append(SpeechChunk(start_sample=start, end_sample=end, audio=piece.copy()))
        if end >= len(audio):
            break
    return chunks or [SpeechChunk(0, len(audio), audio.copy())]


def _group_regions(
    regions: list[tuple[int, int]],
    audio: np.ndarray,
    sample_rate: int,
    config: TranslateLiveConfig,
) -> list[SpeechChunk]:
    min_len = int(config.min_chunk_seconds * sample_rate)
    max_len = int(config.max_chunk_seconds * sample_rate)
    overlap = int(config.overlap_seconds * sample_rate)

    merged: list[tuple[int, int]] = []
    for start, end in regions:
        if end - start < int(sample_rate * 0.15):
            continue
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if end - prev_start <= max_len:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    if not merged:
        return _fixed_windows(audio, sample_rate, config)

    chunks: list[SpeechChunk] = []
    for start, end in merged:
        segment = audio[start:end]
        if len(segment) <= max_len:
            chunks.append(SpeechChunk(start_sample=start, end_sample=end, audio=segment.copy()))
            continue

        step = max(max_len - overlap, 1)
        offset = 0
        while offset < len(segment):
            piece = segment[offset : offset + max_len]
            if len(piece) < min_len and chunks:
                break
            abs_start = start + offset
            abs_end = abs_start + len(piece)
            chunks.append(
                SpeechChunk(start_sample=abs_start, end_sample=abs_end, audio=piece.copy())
            )
            if offset + max_len >= len(segment):
                break
            offset += step

    return chunks or _fixed_windows(audio, sample_rate, config)


def _pad_to_min_length(
    audio: np.ndarray,
    sample_rate: int,
    min_seconds: float,
) -> np.ndarray:
    min_samples = int(min_seconds * sample_rate)
    if len(audio) >= min_samples or len(audio) == 0:
        return audio
    pad = np.zeros(min_samples - len(audio), dtype=audio.dtype)
    return np.concatenate([audio, pad])


def chunk_audio_live(
    audio: np.ndarray,
    sample_rate: int,
    config: TranslateLiveConfig | None = None,
) -> list[SpeechChunk]:
    """Fixed-duration windows for live translation (Gemma STT needs multi-second clips)."""
    from localllm.config import TranslateLiveConfig as LiveCfg

    config = config or LiveCfg()
    if len(audio) == 0:
        return []

    chunk_len = int(config.max_chunk_seconds * sample_rate)
    overlap = int(config.overlap_seconds * sample_rate)
    step = max(chunk_len - overlap, 1)

    chunks: list[SpeechChunk] = []
    for start in range(0, len(audio), step):
        end = min(start + chunk_len, len(audio))
        piece = audio[start:end].copy()
        if end >= len(audio):
            piece = _pad_to_min_length(piece, sample_rate, config.min_chunk_seconds)
        if len(piece) < int(sample_rate * 0.25):
            break
        chunks.append(SpeechChunk(start_sample=start, end_sample=end, audio=piece))
        if end >= len(audio):
            break

    if not chunks:
        piece = _pad_to_min_length(audio.copy(), sample_rate, config.min_chunk_seconds)
        chunks = [SpeechChunk(0, len(audio), piece)]

    return chunks


def chunk_audio_vad(
    audio: np.ndarray,
    sample_rate: int,
    config: TranslateLiveConfig | None = None,
) -> list[SpeechChunk]:
    """Split audio into VAD-guided chunks with overlap.

    Note: the live translate UI currently uses ``chunk_audio_live`` (fixed
    windows); this energy-based VAD is kept for experimentation and is slated
    to be replaced by silero-vad in the streaming pipeline (plan.md, B-2).
    """
    from localllm.config import TranslateLiveConfig as LiveCfg

    config = config or LiveCfg()
    if len(audio) == 0:
        return []

    frame_ms = config.frame_ms
    frame_len = max(int(sample_rate * frame_ms / 1000), 1)
    mask = _speech_mask(
        audio,
        sample_rate,
        frame_ms=frame_ms,
        energy_threshold=config.energy_threshold,
    )
    regions = _mask_to_regions(mask, frame_len, len(audio))
    return _group_regions(regions, audio, sample_rate, config)
