from __future__ import annotations

import base64
import json
import time
from typing import Optional

import httpx

from src.agents.providers.base import ProviderStreamDelta
from src.agents.providers.openai_compatible import OpenAICompatibleProvider
from src.schemas.providers import ProviderModel


_TOKEN_CACHE: dict[str, tuple[str, float]] = {}


class VertexAIProvider(OpenAICompatibleProvider):
    async def list_models(self, api_key: str, base_url: Optional[str] = None) -> list[ProviderModel]:
        token = await self._get_access_token(api_key)
        endpoint = f"{(base_url or self.metadata.default_base_url).rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                endpoint,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
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
        return models or self._fallback_models()

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    async def _get_headers_with_auth(self, api_key: str) -> dict[str, str]:
        token = await self._get_access_token(api_key)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def stream_chat_completion(self, *, api_key, model, messages, tools, base_url=None, temperature=0.2):
        headers = await self._get_headers_with_auth(api_key)
        endpoint = f"{(base_url or self.metadata.default_base_url).rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": temperature,
            "stream": True,
        }

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

    async def _get_access_token(self, api_key: str) -> str:
        cache_entry = _TOKEN_CACHE.get(api_key)
        if cache_entry:
            token, expires_at = cache_entry
            if time.time() < expires_at - 60:
                return token

        try:
            key_data = json.loads(api_key)
            client_email = key_data["client_email"]
            private_key = key_data["private_key"]
            token_uri = key_data.get("token_uri", "https://oauth2.googleapis.com/token")
        except (json.JSONDecodeError, KeyError):
            return api_key

        now = int(time.time())
        header = {"alg": "RS256", "typ": "JWT"}
        claim = {
            "iss": client_email,
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": token_uri,
            "exp": now + 3600,
            "iat": now,
        }

        def _b64url(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

        header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
        claim_b64 = _b64url(json.dumps(claim, separators=(",", ":")).encode())
        signing_input = f"{header_b64}.{claim_b64}"

        signed = self._sign_with_rsa(private_key, signing_input)
        assertion = f"{signing_input}.{signed}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                token_uri,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": assertion,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            data = resp.json()
            access_token = data["access_token"]
            expires_in = data.get("expires_in", 3500)
            _TOKEN_CACHE[api_key] = (access_token, time.time() + expires_in)
            return access_token

    @staticmethod
    def _sign_with_rsa(private_key_pem: str, signing_input: str) -> str:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None,
            backend=default_backend(),
        )
        signature = key.sign(
            signing_input.encode(),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return base64.urlsafe_b64encode(signature).rstrip(b"=").decode()

    def _fallback_models(self) -> list[ProviderModel]:
        model_ids = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]
        return [
            ProviderModel(
                id=mid,
                provider=self.metadata.id,
                label=mid,
            )
            for mid in model_ids
        ]
