"""Memory guard: refuse to download/serve models this machine cannot hold.

The budget is what llama-server can actually use with full GPU offload:
- Apple Silicon (Metal): the GPU working-set cap — ``iogpu.wired_limit_mb``
  when raised, else ~75% of unified memory (macOS's default limit).
- NVIDIA (CUDA): total VRAM across GPUs (``n_gpu_layers: 0`` → CPU budget).
- CPU only: ~80% of system RAM.

Requirements come from the model presets (weights + mmproj + KV/compute
overhead), plus the audio sidecar when it runs alongside the main model.
Override with ``--force`` / ``LOCALLLM_MODEL__SKIP_MEMORY_CHECK=true``.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from dataclasses import dataclass

from localllm.config import AppSettings
from localllm.devices.resolver import detect_platform

GB = 1e9
METAL_DEFAULT_FRACTION = 0.75
CPU_FRACTION = 0.8


class InsufficientMemoryError(RuntimeError):
    """The selected model (plus sidecar) needs more memory than this machine has."""


@dataclass(frozen=True)
class MemoryBudget:
    available_gb: float
    source: str


@dataclass(frozen=True)
class MemoryRequirement:
    total_gb: float
    parts: tuple[tuple[str, float], ...]


def total_ram_bytes() -> int | None:
    if platform.system() == "Windows":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _metal_wired_limit_bytes() -> int | None:
    """User-raised GPU cap (``sudo sysctl iogpu.wired_limit_mb=…``); 0 means default."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        ).stdout.strip()
        mb = int(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return mb * 1024 * 1024 if mb > 0 else None


def _cuda_vram_bytes() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout
        mib = [int(line.strip()) for line in out.splitlines() if line.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return sum(mib) * 1024 * 1024 if mib else None


def memory_budget(settings: AppSettings, *, platform_name: str | None = None) -> MemoryBudget | None:
    """Memory llama-server can use on this machine, or None if undetectable."""
    platform_name = platform_name or detect_platform()
    ram = total_ram_bytes()
    if platform_name == "metal":
        wired = _metal_wired_limit_bytes()
        if wired:
            return MemoryBudget(wired / GB, "GPU limit (iogpu.wired_limit_mb)")
        if ram:
            return MemoryBudget(
                ram * METAL_DEFAULT_FRACTION / GB,
                f"~{METAL_DEFAULT_FRACTION:.0%} of {ram / 2**30:.0f} GB unified memory "
                "(macOS default GPU limit)",
            )
        return None
    if platform_name == "cuda" and settings.llama_server.n_gpu_layers != 0:
        vram = _cuda_vram_bytes()
        if vram:
            return MemoryBudget(vram / GB, "total NVIDIA VRAM")
    if ram:
        return MemoryBudget(ram * CPU_FRACTION / GB, f"~{CPU_FRACTION:.0%} of system RAM")
    return None


def memory_requirement(settings: AppSettings, *, include_sidecar: bool = True) -> MemoryRequirement:
    preset = settings.model.preset()
    parts = [("model", preset.required_memory_gb(settings.model.quantization))]
    if include_sidecar and settings.audio_sidecar_active():
        audio = settings.audio_model_settings().model
        parts.append(
            (f"audio sidecar {audio.name}", audio.preset().required_memory_gb(audio.quantization))
        )
    return MemoryRequirement(sum(gb for _, gb in parts), tuple(parts))


def check_memory_fits(
    settings: AppSettings,
    *,
    include_sidecar: bool = True,
    budget: MemoryBudget | None = None,
) -> MemoryRequirement:
    """Raise InsufficientMemoryError when the model can't fit; returns the requirement."""
    need = memory_requirement(settings, include_sidecar=include_sidecar)
    if settings.model.skip_memory_check:
        return need
    budget = budget or memory_budget(settings)
    if budget is None or need.total_gb <= budget.available_gb:
        return need
    breakdown = " + ".join(f"{name} {gb:.1f} GB" for name, gb in need.parts)
    preset = settings.model.preset()
    others_gb = need.total_gb - need.parts[0][1]
    # Best-quality quantization that would fit.
    smaller = sorted(
        (
            key
            for key in preset.quantizations
            if preset.required_memory_gb(key) + others_gb <= budget.available_gb
        ),
        key=preset.required_memory_gb,
        reverse=True,
    )
    hint = (
        f"Try a smaller quantization: --quant {smaller[0]}."
        if smaller
        else "Pick a smaller model (e.g. gemma-4-12b)."
    )
    raise InsufficientMemoryError(
        f"{preset.label} needs ~{need.total_gb:.1f} GB ({breakdown}) but this machine "
        f"offers ~{budget.available_gb:.1f} GB ({budget.source}). {hint} "
        "To proceed anyway, pass --force or set LOCALLLM_MODEL__SKIP_MEMORY_CHECK=true."
    )
