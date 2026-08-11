from __future__ import annotations

from typing import Any, AsyncGenerator, Optional

import httpx

from src.agents.providers.base import ProviderStreamDelta
from src.agents.providers.openai_compatible import OpenAICompatibleProvider
from src.schemas.providers import ProviderModel


_AI_STUDIO_FALLBACK = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]


class GoogleAIStudioProvider(OpenAICompatibleProvider):
    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

    async def list_models(self, api_key: str, base_url: Optional[str] = None) -> list[ProviderModel]:
        endpoint = f"{(base_url or self.metadata.default_base_url).rstrip('/')}/models"
        headers = self._headers(api_key)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint, headers=headers)
            if response.status_code < 400:
                payload = response.json()
                items = payload.get("data", payload)
                if isinstance(items, list) and items:
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
                    models.sort(key=lambda m: m.label.lower())
                    if models:
                        return models

            return await self._list_via_native_api(api_key)

    async def _list_via_native_api(self, api_key: str) -> list[ProviderModel]:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint)
            if response.status_code >= 400:
                return self._fallback_models()
            payload = response.json()

        items = payload.get("models", [])
        if not items:
            return self._fallback_models()

        models: list[ProviderModel] = []
        for item in items:
            raw_name = item.get("name", "")
            model_id = raw_name.replace("models/", "") if raw_name.startswith("models/") else raw_name
            if not model_id or "embedding" in model_id.lower():
                continue
            label = item.get("displayName") or model_id
            context_window = item.get("inputTokenLimit") or item.get("context_window")
            models.append(
                ProviderModel(
                    id=model_id,
                    provider=self.metadata.id,
                    label=label,
                    owned_by="Google",
                    supports_tools=True,
                    context_window=context_window,
                )
            )

        models.sort(key=lambda m: m.label.lower())
        return models or self._fallback_models()

    async def stream_chat_completion(
        self,
        *,
        api_key: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        base_url: Optional[str] = None,
        temperature: float = 0.2,
    ) -> AsyncGenerator[ProviderStreamDelta, None]:
        endpoint = f"{(base_url or self.metadata.default_base_url).rstrip('/')}/chat/completions"
        headers = self._headers(api_key)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=30.0)) as client:
            async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for event in self._iter_sse_events(response):
                    if event == "[DONE]":
                        break
                    choice = (event.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    finish_reason = choice.get("finish_reason")
                    text = self._extract_text(delta.get("content"))
                    reasoning = self._extract_text(delta.get("reasoning") or delta.get("reasoning_content") or delta.get("reason") or "")
                    tool_calls = delta.get("tool_calls") or None
                    if text or reasoning or tool_calls or finish_reason:
                        yield ProviderStreamDelta(
                            text=text,
                            reasoning=reasoning,
                            tool_calls=tool_calls,
                            finish_reason=finish_reason,
                            raw=event,
                        )

    def _fallback_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                id=mid,
                provider=self.metadata.id,
                label=mid,
            )
            for mid in _AI_STUDIO_FALLBACK
        ]
