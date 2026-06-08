from __future__ import annotations

import atexit
import subprocess
import sys
import time
from typing import TextIO

import httpx

from localllm.config import AppSettings, get_settings


class ServiceManager:
    """Start and manage the FastAPI gateway subprocess."""

    _instance: ServiceManager | None = None

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._proc: subprocess.Popen[str] | None = None
        atexit.register(self.stop)

    @classmethod
    def shared(cls, settings: AppSettings | None = None) -> ServiceManager:
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def _health_url(self) -> str:
        return f"{self.settings.service.base_url.rstrip('/')}/health"

    def is_ready(self, timeout: float = 2.0) -> bool:
        try:
            response = httpx.get(self._health_url(), timeout=timeout)
            if response.status_code != 200:
                return False
            payload = response.json()
            return bool(payload.get("inference_ready", False))
        except (httpx.HTTPError, ValueError):
            return False

    def ensure_running(self, *, wait: bool = True) -> None:
        if self.is_ready():
            return
        if self._proc and self._proc.poll() is None:
            if wait:
                self._wait_ready()
            return

        service = self.settings.service
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "localllm.service.app:app",
            "--host",
            service.host,
            "--port",
            str(service.port),
            "--log-level",
            "warning",
        ]
        print(f"[localllm] Starting LLM gateway on {service.base_url} …")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if wait:
            self._wait_ready()

    def _wait_ready(self) -> None:
        timeout = self.settings.service.startup_timeout_sec
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                output = self._read_process_output(self._proc.stdout)
                raise RuntimeError(
                    f"LLM gateway exited early (code {self._proc.returncode}).\n{output}"
                )
            if self.is_ready(timeout=2.0):
                print("[localllm] LLM gateway is ready.")
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"LLM gateway did not become ready within {timeout}s at "
            f"{self.settings.service.base_url}"
        )

    @staticmethod
    def _read_process_output(stream: TextIO | None) -> str:
        if stream is None:
            return ""
        return stream.read() or ""

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None