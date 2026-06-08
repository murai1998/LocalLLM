from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


def image_part(path: Path) -> dict[str, Any]:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def audio_part(path: Path) -> dict[str, Any]:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "input_audio",
        "input_audio": {"data": b64, "format": path.suffix.lstrip(".") or "wav"},
    }


def text_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}