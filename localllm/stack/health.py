from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ServiceStatus:
    name: str
    url: str
    healthy: bool
    detail: str = ""


def _probe(url: str, *, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        payload = response.json()
        if "inference_ready" in payload:
            ready = bool(payload.get("inference_ready", False))
            return ready, "ready" if ready else str(payload.get("status", "starting"))
        return True, "ok"
    except httpx.HTTPError as exc:
        return False, str(exc)


def service_health(*, llm_base_url: str = "http://127.0.0.1:8090") -> list[ServiceStatus]:
    llm_root = llm_base_url.rstrip("/")
    if llm_root.endswith("/v1"):
        llm_root = llm_root[:-3]

    llm_ok, llm_detail = _probe(f"{llm_root}/health")
    return [ServiceStatus("LocalLLM gateway", llm_root, llm_ok, llm_detail)]