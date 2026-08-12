from __future__ import annotations

from src.agents.providers.openai_compatible import OpenAICompatibleProvider
from src.schemas.providers import ProviderMetadata


class MistralProvider(OpenAICompatibleProvider):
    def __init__(self, metadata: ProviderMetadata) -> None:
        super().__init__(metadata)
