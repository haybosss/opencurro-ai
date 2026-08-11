from __future__ import annotations

from typing import Any

from src.agents.providers.openai_compatible import OpenAICompatibleProvider


class GoogleAIStudioProvider(OpenAICompatibleProvider):
    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
