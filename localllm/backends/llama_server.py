from __future__ import annotations

import atexit
import subprocess
import time
from pathlib import Path

import httpx

from localllm.config import AppSettings, get_settings
from localllm.devices.resolver import detect_platform, llama_server_binary
from localllm.model.download import ensure_gguf_assets


def inference_health_url(settings: AppSettings) -> str:
    return f"{settings.llama_server.base_url.rstrip('/')}/health"


def is_inference_ready(settings: AppSettings, timeout: float = 2.0) -> bool:
    try:
        response = httpx.get(inference_health_url(settings), timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


class LlamaServerManager:
    """Start and manage a local llama-server subprocess."""

    _instance: LlamaServerManager | None = None

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._proc: subprocess.Popen[str] | None = None
        atexit.register(self.stop)

    @classmethod
    def shared(cls, settings: AppSettings | None = None) -> LlamaServerManager:
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def is_ready(self, timeout: float = 2.0) -> bool:
        return is_inference_ready(self.settings, timeout=timeout)

    def build_command(self, gguf: Path, mmproj: Path) -> list[str]:
        srv = self.settings.llama_server
        binary = llama_server_binary()
        cmd = [
            binary,
            "--model",
            str(gguf),
            "--mmproj",
            str(mmproj),
            "--host",
            srv.host,
            "--port",
            str(srv.port),
            "-c",
            str(srv.context_size),
            "-ngl",
            str(srv.n_gpu_layers),
        ]
        if srv.jinja:
            cmd.append("--jinja")
        return cmd

    def start(self, *, wait: bool = True) -> None:
        if self.is_ready():
            return
        if self._proc and self._proc.poll() is None:
            if wait:
                self._wait_ready()
            return

        gguf, mmproj = ensure_gguf_assets(self.settings)
        cmd = self.build_command(gguf, mmproj)
        platform = detect_platform()
        print(f"[localllm] Starting llama-server ({platform}) …")
        print(f"[localllm] Model: {gguf.name}, mmproj: {mmproj.name}")

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if wait:
            self._wait_ready()

    def _wait_ready(self) -> None:
        timeout = self.settings.llama_server.startup_timeout_sec
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                out = ""
                if self._proc.stdout:
                    out = self._proc.stdout.read() or ""
                raise RuntimeError(
                    f"llama-server exited early (code {self._proc.returncode}).\n{out}"
                )
            if self.is_ready():
                print("[localllm] llama-server is ready.")
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"llama-server did not become ready within {timeout}s at "
            f"{self.settings.llama_server.base_url}"
        )

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None