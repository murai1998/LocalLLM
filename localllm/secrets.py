from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_CANDIDATES = (
    ROOT / "hf_token.txt",
    ROOT / ".env",
)


def read_hf_token() -> str | None:
    """Resolve Hugging Face token from env or project files."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env and env.strip():
        return env.strip()

    for path in TOKEN_CANDIDATES:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.startswith("hf_"):
                return line
    return None


def apply_hf_token() -> str | None:
    """Set HF_TOKEN in the environment if found locally."""
    token = read_hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return token