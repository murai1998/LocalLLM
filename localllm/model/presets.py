"""Registry of local models the stack can serve through llama-server.

Each preset names a Hugging Face GGUF repo, the quantizations we ship presets
for, its vision projector (mmproj) and whether the model accepts audio input.
Models without audio (Gemma 4 31B, Qwen3.8-27B) get speech-to-text from an
audio sidecar — see ``AudioSidecarConfig`` in ``localllm.config``.

Sizes are the GGUF weights alone; ``runtime_overhead_gb`` covers the KV cache
and compute buffers at the default 16K context. Together they drive the
memory guard in ``localllm.devices.memory``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class QuantFile:
    file: str
    size_gb: float
    label: str
    # Repo / projector overrides for quants hosted outside the preset's main repo.
    repo: str = ""
    mmproj_file: str = ""
    mmproj_size_gb: float = 0.0


@dataclass(frozen=True)
class ModelPreset:
    id: str
    label: str
    gguf_repo: str
    mmproj_file: str
    quantizations: dict[str, QuantFile]
    default_quantization: str
    # Model id reported by the gateway (and sent by clients).
    served_name: str
    has_audio: bool
    # Subdirectory of model.cache_dir; "" keeps the original flat layout (12B).
    subdir: str
    mmproj_size_gb: float = 0.0
    # KV cache + compute buffers at 16K context (Gemma 31B's 16 KV heads make it heavy).
    runtime_overhead_gb: float = 1.5
    # Extra llama-server flags (sampling defaults the model card recommends).
    server_args: tuple[str, ...] = field(default_factory=tuple)

    def quant(self, quantization: str = "") -> QuantFile:
        key = quantization.strip() or self.default_quantization
        if key not in self.quantizations:
            raise ValueError(
                f"Unknown quantization '{key}' for model '{self.id}'. "
                f"Choose from: {', '.join(self.quantizations)}"
            )
        return self.quantizations[key]

    def required_memory_gb(self, quantization: str = "") -> float:
        """Approximate GPU/unified memory needed to serve this model."""
        quant = self.quant(quantization)
        mmproj = quant.mmproj_size_gb if quant.mmproj_file else self.mmproj_size_gb
        return quant.size_gb + mmproj + self.runtime_overhead_gb


DEFAULT_MODEL: Final = "gemma-4-12b"

PRESETS: Final[dict[str, ModelPreset]] = {
    "gemma-4-12b": ModelPreset(
        id="gemma-4-12b",
        label="Gemma 4 12B (text + image + audio)",
        gguf_repo="unsloth/gemma-4-12b-it-GGUF",
        mmproj_file="mmproj-F16.gguf",
        mmproj_size_gb=0.18,
        runtime_overhead_gb=1.5,
        quantizations={
            "q6_k": QuantFile(
                "gemma-4-12b-it-Q6_K.gguf", 9.8, "6-bit K-quant (default, best quality)"
            ),
            "q5_k": QuantFile(
                "gemma-4-12b-it-Q5_K_M.gguf",
                8.4,
                "5-bit K-quant (smaller; used for the audio sidecar)",
            ),
        },
        default_quantization="q6_k",
        served_name="gemma-4-12b-it",
        has_audio=True,
        subdir="",
    ),
    "gemma-4-31b": ModelPreset(
        id="gemma-4-31b",
        label="Gemma 4 31B (text + image)",
        gguf_repo="google/gemma-4-31B-it-qat-q4_0-gguf",
        mmproj_file="gemma-4-31B-it-mmproj.gguf",
        mmproj_size_gb=1.2,
        runtime_overhead_gb=4.0,
        quantizations={
            # Google's quantization-aware-trained 4-bit: near-BF16 quality at Q4 size.
            "qat_q4_0": QuantFile(
                "gemma-4-31B_q4_0-it.gguf", 17.7, "4-bit QAT (default, official Google)"
            ),
            "q5_k": QuantFile(
                "gemma-4-31B-it-Q5_K_M.gguf",
                21.7,
                "5-bit K-quant (post-training; tight with the audio sidecar on 48 GB)",
                repo="unsloth/gemma-4-31B-it-GGUF",
                mmproj_file="mmproj-F16.gguf",
                mmproj_size_gb=1.2,
            ),
            "q6_k": QuantFile(
                "gemma-4-31B-it-Q6_K.gguf",
                25.2,
                "6-bit K-quant (needs 64 GB, or no audio sidecar)",
                repo="unsloth/gemma-4-31B-it-GGUF",
                mmproj_file="mmproj-F16.gguf",
                mmproj_size_gb=1.2,
            ),
        },
        default_quantization="qat_q4_0",
        served_name="gemma-4-31b-it",
        has_audio=False,
        subdir="gemma-4-31b",
    ),
    "qwen3.8-27b": ModelPreset(
        id="qwen3.8-27b",
        label="Qwen3.8 27B (text + image)",
        gguf_repo="unsloth/Qwen3.8-27B-GGUF",
        mmproj_file="mmproj-F16.gguf",
        mmproj_size_gb=0.93,
        runtime_overhead_gb=2.0,
        quantizations={
            "ud_q5_k_xl": QuantFile(
                "Qwen3.8-27B-UD-Q5_K_XL.gguf", 20.9, "5-bit Unsloth Dynamic (default)"
            ),
            "ud_q4_k_xl": QuantFile(
                "Qwen3.8-27B-UD-Q4_K_XL.gguf", 17.6, "4-bit Unsloth Dynamic (smaller)"
            ),
            "ud_q6_k": QuantFile("Qwen3.8-27B-UD-Q6_K.gguf", 22.0, "6-bit Unsloth Dynamic"),
        },
        default_quantization="ud_q5_k_xl",
        served_name="qwen3.8-27b",
        has_audio=False,
        subdir="qwen3.8-27b",
        # Model-card defaults; per-request temperature/top_p still win.
        server_args=("--top-k", "20", "--min-p", "0.0"),
    ),
}


def get_preset(name: str = "") -> ModelPreset:
    key = name.strip() or DEFAULT_MODEL
    if key not in PRESETS:
        raise ValueError(f"Unknown model '{key}'. Choose from: {', '.join(PRESETS)}")
    return PRESETS[key]


def list_models() -> list[str]:
    return list(PRESETS)
