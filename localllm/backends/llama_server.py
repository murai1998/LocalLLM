from __future__ import annotations

import atexit
import subprocess
import time
from pathlib import Path

import httpx

from localllm.config import ROOT, AppSettings, get_settings
from localllm.devices.resolver import detect_platform, llama_server_binary
from localllm.model.download import ensure_gguf_assets

LOG_DIR = ROOT / "logs"


def inference_health_url(settings: AppSettings) -> str:
    return f"{settings.llama_server.base_url.rstrip('/')}/health"


def is_inference_ready(settings: AppSettings, timeout: float = 2.0) -> bool:
    try:
        response = httpx.get(inference_health_url(settings), timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


class LlamaServerManager:
    """Start and manage a local llama-server subprocess.

    ``role`` distinguishes the main model ("primary") from the audio sidecar
    ("audio"); each role is one process with its own settings view.
    """

    _instances: dict[str, LlamaServerManager] = {}

    def __init__(self, settings: AppSettings | None = None, *, role: str = "primary") -> None:
        self.settings = settings or get_settings()
        self.role = role
        self._proc: subprocess.Popen[str] | None = None
        atexit.register(self.stop)

    @classmethod
    def shared(cls, settings: AppSettings | None = None, *, role: str = "primary") -> LlamaServerManager:
        if role not in cls._instances:
            cls._instances[role] = cls(settings, role=role)
        return cls._instances[role]

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"llama-server-{self.role}.log"

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
        cmd.extend(["--alias", self.settings.model.preset().served_name])
        cmd.extend(self.settings.model.preset().server_args)
        return cmd

    def served_model_ids(self) -> list[str]:
        try:
            response = httpx.get(f"{self.settings.llama_server.base_url}/v1/models", timeout=2.0)
            response.raise_for_status()
            return [m.get("id", "") for m in response.json().get("data", [])]
        except (httpx.HTTPError, ValueError):
            return []

    def start(self, *, wait: bool = True) -> None:
        if self.is_ready():
            expected = self.settings.model.preset().served_name
            served = self.served_model_ids()
            if served and expected not in served:
                raise RuntimeError(
                    f"{self.settings.llama_server.base_url} is already serving {served}, "
                    f"not '{expected}'. Stop that llama-server first "
                    "(scripts/stop_stale_ports.sh) and retry."
                )
            return
        if self._proc and self._proc.poll() is None:
            if wait:
                self._wait_ready()
            return

        gguf, mmproj = ensure_gguf_assets(self.settings)
        cmd = self.build_command(gguf, mmproj)
        platform = detect_platform()
        url = self.settings.llama_server.base_url
        print(f"[localllm] Starting llama-server [{self.role}] ({platform}) on {url} …")
        print(f"[localllm] Model: {gguf.name}, mmproj: {mmproj.name}")
        print(f"[localllm] Log: {self.log_path}")

        # Log to a file: an unread PIPE fills up and blocks llama-server mid-request.
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as log:
            self._proc = subprocess.Popen(
                cmd,
                stdout=log,
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
                raise RuntimeError(
                    f"llama-server [{self.role}] exited early (code {self._proc.returncode}).\n"
                    f"{self._log_tail()}"
                )
            if self.is_ready():
                print("[localllm] llama-server is ready.")
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"llama-server did not become ready within {timeout}s at "
            f"{self.settings.llama_server.base_url}"
        )

    def _log_tail(self, lines: int = 40) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None