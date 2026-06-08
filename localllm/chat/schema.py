from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UserTurn:
    text: str = ""
    image_paths: list[Path] = field(default_factory=list)
    audio_path: Path | None = None

    def has_media(self) -> bool:
        return bool(self.image_paths) or self.audio_path is not None


@dataclass
class ChatMessage:
    role: str
    content: str | list[dict[str, Any]]

    def to_api(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}