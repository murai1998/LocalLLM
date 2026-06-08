from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """OpenAI-compatible chat client used by all LocalLLM apps."""

    def is_ready(self, timeout: float = 2.0) -> bool:
        """Return True when the configured endpoint is reachable."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Send a chat completion request and return assistant text."""

    def image_part(self, path: Path) -> dict[str, Any]: ...

    def text_part(self, text: str) -> dict[str, str]: ...

    def audio_part(self, path: Path) -> dict[str, Any]: ...