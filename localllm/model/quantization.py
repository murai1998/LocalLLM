from __future__ import annotations

from typing import Final

QUANT_PRESETS: Final[dict[str, dict[str, str]]] = {
    "q6_k": {
        "gguf_file": "gemma-4-12b-it-Q6_K.gguf",
        "label": "6-bit K-quant (default, best quality)",
    },
    "q5_k": {
        "gguf_file": "gemma-4-12b-it-Q5_K_M.gguf",
        "label": "5-bit K-quant (smaller VRAM, use with Whisper on GPU)",
    },
}

DEFAULT_QUANTIZATION = "q6_k"


def resolve_gguf_file(*, quantization: str, gguf_file: str = "") -> str:
    """Resolve GGUF filename from quantization preset or explicit override."""
    if gguf_file.strip():
        return gguf_file.strip()
    preset = QUANT_PRESETS.get(quantization)
    if preset is None:
        raise ValueError(
            f"Unknown quantization '{quantization}'. "
            f"Choose from: {', '.join(QUANT_PRESETS)}"
        )
    return preset["gguf_file"]


def list_quantizations() -> list[str]:
    return list(QUANT_PRESETS)