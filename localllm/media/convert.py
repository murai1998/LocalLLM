from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from localllm.config import ROOT

UPLOAD_CACHE_ROOT = Path(tempfile.gettempdir()) / "localllm"

AUDIO_INPUT_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".opus"}
DOC_EXTRACT_SUFFIXES = {".txt", ".md", ".docx", ".csv", ".json", ".yaml", ".yml", ".html", ".htm", ".log", ".xml"}
COMPRESSED_AUDIO_SUFFIXES = AUDIO_INPUT_SUFFIXES - {".wav"}


def resolve_ffmpeg_executable() -> str | None:
    """System ffmpeg on PATH, or bundled binary from imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffmpeg_available() -> bool:
    return resolve_ffmpeg_executable() is not None


def safe_media_path(path: str | Path) -> Path:
    """Resolve a path under the project root or streamlit upload cache."""
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    allowed_roots = (ROOT.resolve(), UPLOAD_CACHE_ROOT.resolve())
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        "Path must stay inside the project directory or LocalLLM upload cache."
    )


def _ffmpeg_to_wav(source: Path, dest: Path, *, sample_rate: int = 16000) -> Path:
    ffmpeg = resolve_ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available")

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
        raise RuntimeError(stderr[-500:])
    if not dest.is_file():
        raise RuntimeError("ffmpeg did not produce an output file")
    return dest


def convert_audio_to_wav(
    source: Path,
    *,
    out_dir: Path | None = None,
    sample_rate: int = 16000,
) -> Path:
    """Convert readable audio to 16 kHz mono WAV (ffmpeg preferred for m4a/mp3)."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")

    suffix = source.suffix.lower()
    if suffix not in AUDIO_INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported audio type '{suffix}'. "
            f"Supported: {', '.join(sorted(AUDIO_INPUT_SUFFIXES))}"
        )

    out_dir = out_dir or Path(tempfile.gettempdir()) / "localllm" / "converted"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{source.stem}_16k.wav"

    if suffix == ".wav":
        from localllm.media.audio import load_mono_16k, write_wav

        audio = load_mono_16k(source, sample_rate=sample_rate)
        return write_wav(dest, audio, sample_rate=sample_rate)

    if ffmpeg_available():
        return _ffmpeg_to_wav(source, dest, sample_rate=sample_rate)

    if suffix in COMPRESSED_AUDIO_SUFFIXES:
        raise RuntimeError(
            f"Cannot convert {suffix} without ffmpeg. "
            "Install `imageio-ffmpeg` (pip) or system ffmpeg on PATH, or upload WAV."
        )

    from localllm.media.audio import load_mono_16k, write_wav

    audio = load_mono_16k(source, sample_rate=sample_rate)
    return write_wav(dest, audio, sample_rate=sample_rate)