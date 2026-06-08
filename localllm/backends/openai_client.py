from __future__ import annotations

from localllm.client.openai_compatible import OpenAICompatibleClient


class LlamaOpenAIClient(OpenAICompatibleClient):
    """Backward-compatible alias for the shared OpenAI-compatible client."""