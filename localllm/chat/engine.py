from __future__ import annotations

from pathlib import Path
from typing import Any

from localllm.chat.schema import ChatMessage, UserTurn
from localllm.client.factory import create_llm_client
from localllm.client.protocol import LLMClient
from localllm.config import AppSettings, get_settings
from localllm.service.manager import ServiceManager


class ChatEngine:
    """Multi-turn chat via the configured LLM client (local gateway or commercial API)."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: AppSettings | None = None,
        *,
        system_prompt: str = "You are a helpful assistant.",
        autostart_server: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or create_llm_client(self.settings)
        self.system_prompt = system_prompt
        self.history: list[dict[str, Any]] = []
        if autostart_server and self.settings.llm.provider == "local":
            ServiceManager.shared(self.settings).ensure_running()

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def reset(self, system_prompt: str | None = None) -> None:
        prompt = system_prompt if system_prompt is not None else self.system_prompt
        self.history = []
        if prompt:
            self.history.append({"role": "system", "content": prompt})

    def _build_user_content(self, turn: UserTurn) -> str | list[dict[str, Any]]:
        if not turn.has_media():
            return turn.text

        parts: list[dict[str, Any]] = []
        for img in turn.image_paths:
            parts.append(self.client.image_part(img))
        if turn.text:
            parts.append(self.client.text_part(turn.text))
        if turn.audio_path:
            parts.append(self.client.audio_part(turn.audio_path))
        return parts

    def send(self, turn: UserTurn | str, *, max_tokens: int | None = None) -> str:
        if isinstance(turn, str):
            turn = UserTurn(text=turn)

        user_content = self._build_user_content(turn)
        self.history.append({"role": "user", "content": user_content})

        reply = self.client.chat(self.history, max_tokens=max_tokens)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def send_with_images(
        self,
        text: str,
        image_paths: list[Path],
        *,
        max_tokens: int | None = None,
    ) -> str:
        return self.send(UserTurn(text=text, image_paths=image_paths), max_tokens=max_tokens)

    def send_with_audio(
        self,
        text: str,
        audio_path: Path,
        *,
        max_tokens: int | None = None,
    ) -> str:
        return self.send(
            UserTurn(text=text, audio_path=audio_path),
            max_tokens=max_tokens,
        )

    @property
    def messages(self) -> list[ChatMessage]:
        return [ChatMessage(role=m["role"], content=m["content"]) for m in self.history]