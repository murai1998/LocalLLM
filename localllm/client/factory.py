from __future__ import annotations

from localllm.client.openai_compatible import OpenAICompatibleClient
from localllm.client.protocol import LLMClient
from localllm.config import AppSettings, get_settings


def create_llm_client(settings: AppSettings | None = None) -> LLMClient:
    """Build the configured LLM client (local gateway or commercial API)."""
    return OpenAICompatibleClient(settings or get_settings())