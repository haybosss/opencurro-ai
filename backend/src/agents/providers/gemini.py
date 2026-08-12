from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Optional

import httpx

from src.agents.providers.base import LLMProvider, ProviderStreamDelta
from src.schemas.providers import ProviderMetadata, ProviderModel


class GeminiProvider(LLMProvider):
    _FALLBACK_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3.0-flash",
        "gemini-3.0-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
    ]

    def __init__(self, metadata: ProviderMetadata) -> None:
        self.metadata = metadata
        self._models_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
        self._generate_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"

    async def list_models(self, api_key: str, base_url: Optional[str] = None) -> list[ProviderModel]:
        endpoint = f"{self._models_endpoint}?key={api_key}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint)
            if response.status_code >= 400:
                return self._fallback_models()
            payload = response.json()

        items = payload.get("models", [])
        models: list[ProviderModel] = []
        for item in items:
            name = item.get("name", "")
            model_id = name.split("/")[-1] if name else ""
            if not model_id:
                continue
            supported_methods = item.get("supportedGenerationMethods", [])
            if "streamGenerateContent" not in supported_methods:
                continue

            input_limit = item.get("inputTokenLimit")
            output_limit = item.get("outputTokenLimit")
            context_window = None
            if input_limit is not None and output_limit is not None:
                context_window = input_limit + output_limit
            elif input_limit is not None:
                context_window = input_limit

            models.append(
                ProviderModel(
                    id=model_id,
                    provider=self.metadata.id,
                    label=item.get("displayName") or model_id,
                    owned_by="Google",
                    supports_tools="generateContent" in supported_methods,
                    context_window=context_window,
                )
            )
        models.sort(key=lambda model: model.label.lower())
        if not models:
            return self._fallback_models()
        return models

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
        endpoint = self._generate_endpoint.format(model=model)
        endpoint += f"?alt=sse&key={api_key}"

        contents, system_instruction = self._convert_messages(messages)
        function_declarations = self._convert_tools(tools)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if function_declarations:
            payload["tools"] = [{"functionDeclarations": function_declarations}]

        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=30.0)) as client:
            async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                await self._handle_api_error(response)
                async for event in self._iter_sse_events(response):
                    if event == "[DONE]":
                        break
                    delta = self._parse_gemini_event(event)
                    if delta:
                        yield delta

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            if role == "system":
                if content:
                    system_parts.append({"text": content})
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                content_str = content if isinstance(content, str) else json.dumps(content) if content else ""
                parts = [{
                    "functionResponse": {
                        "name": tool_call_id,
                        "response": {"result": content_str},
                    }
                }]
                contents.append({"role": "user", "parts": parts})
                continue

            parts: list[dict[str, Any]] = []
            if content:
                if isinstance(content, str):
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif isinstance(item, dict):
                            parts.append(item)
                        else:
                            parts.append({"text": str(item)})
                else:
                    parts.append({"text": str(content)})

            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    parts.append({
                        "functionCall": {
                            "name": func.get("name", ""),
                            "args": args,
                        }
                    })

            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": parts})

        system_instruction = None
        if system_parts:
            system_instruction = {"parts": system_parts}

        return contents, system_instruction

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            gemini_params: dict[str, Any] = {"type": parameters.get("type", "object")}
            if "properties" in parameters:
                gemini_params["properties"] = parameters["properties"]
            if "required" in parameters:
                gemini_params["required"] = parameters["required"]

            declarations.append({
                "name": name,
                "description": description,
                "parameters": gemini_params,
            })
        return declarations

    def _parse_gemini_event(self, event: dict[str, Any]) -> Optional[ProviderStreamDelta]:
        candidates = event.get("candidates") or []
        if not candidates:
            return None

        candidate = candidates[0]
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        finish_reason = candidate.get("finishReason")

        text = ""
        reasoning = ""
        tool_calls: list[dict[str, Any]] = []

        for i, part in enumerate(parts):
            if "text" in part:
                text += part.get("text", "")
            if "thought" in part:
                reasoning += str(part.get("thought", ""))
            if "functionCall" in part:
                fc = part["functionCall"]
                args = fc.get("args", {})
                args_str = json.dumps(args)

                tool_calls.append({
                    "index": i,
                    "id": fc.get("name", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": args_str,
                    },
                })

        finish_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "MALFORMED_FUNCTION_CALL": "tool_calls",
        }
        mapped_finish = finish_map.get(finish_reason or "", finish_reason)

        has_content = bool(text or reasoning or tool_calls)
        if has_content or mapped_finish:
            return ProviderStreamDelta(
                text=text,
                reasoning=reasoning,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=mapped_finish,
                raw=event,
            )
        return None

    async def _handle_api_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            body = await response.aread()
            payload = json.loads(body)
            err_detail = payload.get("error", {}).get("message", body.decode(errors="replace"))
        except Exception:
            err_detail = ""
        if response.status_code == 404:
            err_msg = (
                "Gemini model not found (404). The model may not be available for your API key "
                "or may have been deprecated. Try using a different model."
            )
            if err_detail:
                err_msg += f" Details: {err_detail}"
        elif response.status_code == 429:
            err_msg = "Gemini rate limit exceeded (429). Please wait and try again."
            if err_detail:
                err_msg += f" Details: {err_detail}"
        elif response.status_code == 403:
            err_msg = (
                "Gemini access denied (403). Your API key may not have access to this model "
                "or your account may have restrictions."
            )
            if err_detail:
                err_msg += f" Details: {err_detail}"
        else:
            err_msg = f"Gemini API error ({response.status_code})"
            if err_detail:
                err_msg += f": {err_detail}"
        raise httpx.HTTPStatusError(err_msg, request=response.request, response=response)

    def _fallback_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                id=mid,
                provider=self.metadata.id,
                label=mid,
                owned_by="Google",
                supports_tools=True,
            )
            for mid in self._FALLBACK_MODELS
        ]

    async def _iter_sse_events(self, response: httpx.Response) -> AsyncGenerator[dict[str, Any] | str, None]:
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                raw_event, buffer = buffer.split("\n\n", 1)
                for event in self._parse_sse_raw(raw_event):
                    yield event
        if buffer.strip():
            for event in self._parse_sse_raw(buffer.strip()):
                yield event

    def _parse_sse_raw(self, raw: str) -> list[dict[str, Any] | str]:
        lines = raw.split("\n")
        data_parts: list[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_parts.append(line[5:].strip())
        if not data_parts:
            return []
        data = "\n".join(data_parts)
        if data == "[DONE]":
            return [data]
        try:
            return [json.loads(data)]
        except json.JSONDecodeError:
            return []
