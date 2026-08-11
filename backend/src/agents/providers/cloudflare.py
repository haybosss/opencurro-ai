from __future__ import annotations

import re
from typing import Any, AsyncGenerator, Optional

import httpx

from src.agents.providers.base import ProviderStreamDelta
from src.agents.providers.openai_compatible import OpenAICompatibleProvider
from src.schemas.providers import ProviderModel


_CF_MODEL_IDS = [
    "@cf/meta/llama-4-maverick-17b-128e-instruct",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-3.1-8b-instruct",
    "@cf/meta/llama-3.1-70b-instruct",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    "@cf/qwen/qwen3-235b-a22b-instruct-2507",
    "@cf/qwen/qwq-32b",
    "@cf/qwen/qwen2.5-coder-32b-instruct",
    "@cf/google/gemma-3-27b-it",
    "@cf/mistral/mistral-large-2-123b",
    "@cf/mistral/mistral-small-3.1-24b",
    "@cf/mistral/codestral-2501",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
]


def _norm_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    url = re.sub(r"/(chat/completions|completions)$", "", url)
    return url


def _extract_account_id(url: str) -> Optional[str]:
    m = re.search(r"/accounts/([a-f0-9]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"gateway\.ai\.cloudflare\.com/v1/([a-f0-9]+)", url)
    if m:
        return m.group(1)
    return None


class CloudflareProvider(OpenAICompatibleProvider):
    async def list_models(self, api_key: str, base_url: Optional[str] = None) -> list[ProviderModel]:
        base = _norm_base_url(base_url or self.metadata.default_base_url)
        endpoint = f"{base}/models"
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
                    if models:
                        models.sort(key=lambda m: m.label.lower())
                        return models

        account_id = _extract_account_id(base)
        if account_id:
            return await self._list_via_workers_api(api_key, account_id)

        return self._fallback_models()

    async def _list_via_workers_api(self, api_key: str, account_id: str) -> list[ProviderModel]:
        endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint, headers=headers)
            if response.status_code >= 400:
                return self._fallback_models()
            payload = response.json()

        items = payload.get("result", [])
        if not isinstance(items, list) or not items:
            return self._fallback_models()

        models: list[ProviderModel] = []
        for item in items:
            model_id = item.get("name") or item.get("id") or ""
            if not model_id or "embedding" in model_id.lower():
                continue
            label = item.get("name") or model_id
            context_window = item.get("context_window") or item.get("context_length")
            models.append(
                ProviderModel(
                    id=model_id,
                    provider=self.metadata.id,
                    label=label,
                    owned_by=item.get("source") or item.get("provider") or "Cloudflare",
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
        base = _norm_base_url(base_url or self.metadata.default_base_url)
        endpoint = f"{base}/chat/completions"
        headers = self._headers(api_key)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

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
            for mid in _CF_MODEL_IDS
        ]
