from __future__ import annotations

from pathlib import Path

from PIL import Image


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def ensure_local_image(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not is_image(path):
        raise ValueError(f"Unsupported image type: {path.suffix}")
    with Image.open(path) as img:
        img.verify()
    return path