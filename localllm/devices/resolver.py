from __future__ import annotations

import platform
import shutil
import subprocess


def detect_platform() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "metal"
    if system == "windows":
        return "cuda" if _has_nvidia_smi() else "cpu"
    return "cuda" if _has_nvidia_smi() else "cpu"


def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None and (
        subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def llama_server_binary() -> str:
    """Resolve llama-server executable (PATH or common names)."""
    for name in ("llama-server", "llama-server.exe", "server"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "llama-server not found on PATH. Install llama.cpp from "
        "https://github.com/ggml-org/llama.cpp/releases and add llama-server to PATH."
    )