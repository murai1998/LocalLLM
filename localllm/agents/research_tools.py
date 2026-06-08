from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
from langchain_core.tools import tool

from localllm.agents.tools import _safe_path
from localllm.config import ROOT, get_settings
from localllm.devices.resolver import detect_platform


_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".sh",
    ".ps1",
    ".bat",
    ".sql",
    ".rst",
    ".csv",
    ".xml",
}


@tool
def search_project(
    query: str,
    path: str = ".",
    max_results: int = 20,
) -> str:
    """Search filenames and text contents inside the project."""
    root = _safe_path(path)
    if not root.is_dir():
        return f"Error: not a directory: {path}"

    needle = query.strip().lower()
    if not needle:
        return "Error: empty search query."

    max_results = max(1, min(int(max_results), 50))
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    matches: list[dict[str, str | int]] = []

    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        if any(part.startswith(".") for part in candidate.relative_to(root).parts):
            continue
        if candidate.suffix.lower() not in _TEXT_SUFFIXES and candidate.name not in {
            "Dockerfile",
            "Makefile",
        }:
            continue

        rel = candidate.relative_to(ROOT).as_posix()
        name_hit = needle in candidate.name.lower()
        line_hit = None
        if not name_hit:
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    line_hit = line_no
                    snippet = line.strip()
                    break
        else:
            snippet = "(filename match)"

        if name_hit or line_hit is not None:
            matches.append(
                {
                    "path": rel,
                    "line": line_hit or 0,
                    "snippet": snippet[:200],
                }
            )
        if len(matches) >= max_results:
            break

    if not matches:
        return json.dumps({"query": query, "matches": [], "note": "No matches found."})
    return json.dumps({"query": query, "matches": matches}, ensure_ascii=False)


def _now_payload() -> dict[str, object]:
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    return {
        "local": {
            "iso": now_local.isoformat(),
            "date": now_local.strftime("%Y-%m-%d"),
            "time": now_local.strftime("%H:%M:%S"),
            "weekday": now_local.strftime("%A"),
            "timezone": str(now_local.tzinfo),
        },
        "utc": {
            "iso": now_utc.isoformat(),
            "date": now_utc.strftime("%Y-%m-%d"),
            "time": now_utc.strftime("%H:%M:%S"),
            "weekday": now_utc.strftime("%A"),
        },
    }


@tool
def get_current_datetime() -> str:
    """Return the current local date, time, and weekday from the system clock."""
    return json.dumps(_now_payload(), ensure_ascii=False, indent=2)


@tool
def get_system_status() -> str:
    """Report platform, gateway health, inference backend readiness, and current time."""
    settings = get_settings()
    gateway_url = f"{settings.service.base_url.rstrip('/')}/health"
    inference_url = f"{settings.llama_server.base_url.rstrip('/')}/health"

    gateway: dict[str, object] = {"url": gateway_url, "ready": False}
    inference: dict[str, object] = {"url": inference_url, "ready": False}

    try:
        response = httpx.get(gateway_url, timeout=3.0)
        gateway["ready"] = response.status_code == 200
        if response.status_code == 200:
            gateway.update(response.json())
    except httpx.HTTPError as exc:
        gateway["error"] = str(exc)

    try:
        response = httpx.get(inference_url, timeout=3.0)
        inference["ready"] = response.status_code == 200
    except httpx.HTTPError as exc:
        inference["error"] = str(exc)

    clock = _now_payload()
    return json.dumps(
        {
            "platform": detect_platform(),
            "datetime": clock,
            "timestamp_utc": clock["utc"]["iso"],
            "project_root": str(ROOT),
            "gateway": gateway,
            "inference": inference,
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
        },
        ensure_ascii=False,
        indent=2,
    )