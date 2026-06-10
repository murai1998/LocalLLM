from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from localllm.config import SttConfig


def load_mono_16k(path: Path, sample_rate: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    sf.write(path, pcm, sample_rate, subtype="PCM_16")
    return path


def to_wav_16k(source: Path, out_dir: Path | None = None) -> Path:
    """Convert any readable audio file to 16 kHz mono WAV for Gemma."""
    from localllm.media.convert import convert_audio_to_wav

    return convert_audio_to_wav(source, out_dir=out_dir)


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: SttConfig,
) -> list[np.ndarray]:
    """Split audio into overlapping chunks (max 30s per Gemma limit)."""
    chunk_len = int(min(config.chunk_seconds, config.max_chunk_seconds) * sample_rate)
    max_len = int(config.max_chunk_seconds * sample_rate)
    overlap = int(config.overlap_seconds * sample_rate)
    step = max(chunk_len - overlap, 1)
    chunks: list[np.ndarray] = []
    for start in range(0, len(audio), step):
        piece = audio[start : start + chunk_len]
        if len(piece) < sample_rate * 0.5:
            # Don't drop a short trailing piece: extend the previous chunk to the
            # end of the audio when it still fits under the hard Gemma limit,
            # otherwise keep the tail as its own small chunk.
            if chunks:
                prev_start = start - step
                extended = audio[prev_start : start + len(piece)]
                if len(extended) <= max_len:
                    chunks[-1] = extended
                elif len(piece):
                    chunks.append(piece)
            break
        chunks.append(piece)
        if start + chunk_len >= len(audio):
            break
    return chunks or [audio[:chunk_len]]


def merge_transcripts(parts: list[str]) -> str:
    """Merge chunk transcripts with simple overlap deduplication."""
    if not parts:
        return ""
    merged = parts[0].strip()
    for nxt in parts[1:]:
        nxt = nxt.strip()
        if not nxt:
            continue
        overlap = _longest_suffix_prefix_overlap(merged, nxt)
        if overlap > 8:
            merged += nxt[overlap:]
        else:
            merged += " " + nxt
    return merged.strip()


def _longest_suffix_prefix_overlap(a: str, b: str, max_check: int = 200) -> int:
    max_len = min(len(a), len(b), max_check)
    for size in range(max_len, 0, -1):
        if a[-size:] == b[:size]:
            return size
    return 0
