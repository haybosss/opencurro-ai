from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Optional

import httpx

from src.agents.providers.base import LLMProvider, ProviderStreamDelta
from src.schemas.providers import ProviderMetadata, ProviderModel


class GeminiProvider(LLMProvider):
    def __init__(self, metadata: ProviderMetadata) -> None:
        self.metadata = metadata
        self._models_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
        self._generate_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"

    async def list_models(self, api_key: str, base_url: Optional[str] = None) -> list[ProviderModel]:
        endpoint = f"{self._models_endpoint}?key={api_key}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            payload = response.json()

        items = payload.get("models", [])
        models: list[ProviderModel] = []
        for item in items:
            name = item.get("name", "")
            model_id = name.replace("models/", "")
            if not model_id:
                continue
            supported_methods = item.get("supportedGenerationMethods", [])

            supports_tools = "generateContent" in supported_methods
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
                    supports_tools=supports_tools,
                    context_window=context_window,
                )
            )
        models.sort(key=lambda model: model.label.lower())
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
                response.raise_for_status()
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
                text += part["text"]
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

    async def _iter_sse_events(self, response: httpx.Response) -> AsyncGenerator[dict[str, Any] | str, None]:
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                raw_event, buffer = buffer.split("\n\n", 1)
                data_lines: list[str] = []
                for line in raw_event.splitlines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                if not data_lines:
                    continue
                data = "\n".join(data_lines)
                if data == "[DONE]":
                    yield data
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
