from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from localllm.client import media
from localllm.config import AppSettings, get_settings


class OpenAICompatibleClient:
    """HTTP client for any OpenAI-compatible chat API (local gateway or commercial)."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        llm = self.settings.llm
        self.base_url = llm.base_url.rstrip("/")
        self.api_key = llm.api_key or os.environ.get("OPENAI_API_KEY")
        self.model = llm.model
        self.timeout = llm.timeout_sec

    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _health_url(self) -> str:
        root = self.base_url
        if root.endswith("/v1"):
            root = root[:-3]
        return f"{root.rstrip('/')}/health"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def is_ready(self, timeout: float = 2.0) -> bool:
        try:
            response = httpx.get(self._health_url(), timeout=timeout, headers=self._headers())
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        gen = self.settings.generation
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else gen.max_tokens,
            "temperature": temperature if temperature is not None else gen.temperature,
            "top_p": top_p if top_p is not None else gen.top_p,
        }
        if self.settings.llm.provider == "local":
            body["chat_template_kwargs"] = {
                "enable_thinking": (
                    enable_thinking
                    if enable_thinking is not None
                    else gen.enable_thinking
                ),
            }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self._chat_url(),
                json=body,
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]["message"]
        finish = data["choices"][0].get("finish_reason", "")
        content = (choice.get("content") or "").strip()
        reasoning = (choice.get("reasoning_content") or "").strip()

        if content:
            if finish == "length":
                content += "\n\n[Response truncated — increase generation.max_tokens in config.]"
            return content

        if reasoning:
            if finish == "length":
                reasoning += "\n\n[Response truncated — increase generation.max_tokens in config.]"
            return reasoning

        return ""

    def image_part(self, path: Path) -> dict[str, Any]:
        return media.image_part(path)

    def text_part(self, text: str) -> dict[str, str]:
        return media.text_part(text)

    def audio_part(self, path: Path) -> dict[str, Any]:
        return media.audio_part(path)