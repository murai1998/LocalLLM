from __future__ import annotations

from typing import Final

from localllm.model.presets import DEFAULT_MODEL, get_preset

# Gemma 4 12B quant table, kept for callers that predate model presets.
QUANT_PRESETS: Final[dict[str, dict[str, str]]] = {
    key: {"gguf_file": q.file, "label": q.label}
    for key, q in get_preset(DEFAULT_MODEL).quantizations.items()
}

DEFAULT_QUANTIZATION = get_preset(DEFAULT_MODEL).default_quantization


def resolve_gguf_file(*, quantization: str, gguf_file: str = "", model: str = "") -> str:
    """Resolve GGUF filename from the model preset's quantization or an explicit override."""
    if gguf_file.strip():
        return gguf_file.strip()
    return get_preset(model).quant(quantization).file


def list_quantizations(model: str = "") -> list[str]:
    return list(get_preset(model).quantizations)
