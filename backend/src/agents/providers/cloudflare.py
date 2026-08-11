from __future__ import annotations

from typing import Optional

import httpx

from src.agents.providers.openai_compatible import OpenAICompatibleProvider
from src.schemas.providers import ProviderModel


_CLOUDFLARE_FALLBACK_MODELS = [
    "@cf/meta/llama-4-maverick-17b-128e-instruct",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-3.1-8b-instruct",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    "@cf/qwen/qwen3-235b-a22b-instruct-2507",
    "@cf/qwen/qwq-32b",
    "@cf/google/gemma-3-27b-it",
    "@cf/mistral/mistral-large-2-123b",
    "@cf/mistral/mistral-small-3.1-24b",
]


class CloudflareProvider(OpenAICompatibleProvider):
    async def list_models(self, api_key: str, base_url: Optional[str] = None) -> list[ProviderModel]:
        endpoint = f"{(base_url or self.metadata.default_base_url).rstrip('/')}/models"
        headers = self._headers(api_key)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint, headers=headers)
            if response.status_code >= 400:
                return self._fallback_models()
            payload = response.json()

        items = payload.get("data", payload)
        if not isinstance(items, list) or not items:
            return self._fallback_models()

        models: list[ProviderModel] = []
        for item in items:
            model_id = item.get("id") or item.get("name")
            if not model_id:
                continue
            models.append(
                ProviderModel(
                    id=model_id,
                    provider=self.metadata.id,
                    label=model_id,
                    owned_by=item.get("owned_by"),
                    supports_tools=True,
                )
            )
        models.sort(key=lambda model: model.label.lower())
        return models

    def _fallback_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                id=mid,
                provider=self.metadata.id,
                label=mid,
            )
            for mid in _CLOUDFLARE_FALLBACK_MODELS
        ]
