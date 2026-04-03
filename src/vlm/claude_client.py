"""VLM client via OpenRouter API (OpenAI-compatible with vision + tool use).

Works with any model on OpenRouter that supports vision and tool_use:
- anthropic/claude-sonnet-4
- qwen/qwen-2.5-vl-72b-instruct
- google/gemini-2.5-flash
"""
from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path

import httpx

from .base import VLMClient, VLMResponse, ToolCall


class OpenRouterVLMClient(VLMClient):
    """VLM client using OpenRouter's OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str = "anthropic/claude-sonnet-4",
                 base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=90.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._conversation: list[dict] = []

    def analyze(self, image_path: str, prompt: str,
                tools: list[dict] | None = None,
                system: str = "") -> VLMResponse:
        """Send image + prompt to VLM via OpenRouter."""
        content = []

        # Add image (resize if too large to avoid token/timeout issues)
        if image_path and Path(image_path).exists():
            img_data = self._prepare_image(image_path)
            media_type = "image/png"
            b64 = base64.standard_b64encode(img_data).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            })

        # Input size guard: truncate prompt if too large for context window
        max_prompt_chars = 600_000  # ~150K tokens, safe for 200K context
        if len(prompt) > max_prompt_chars:
            half = max_prompt_chars // 2
            prompt = (
                prompt[:half]
                + f"\n\n[... {len(prompt) - max_prompt_chars:,} chars truncated for context limit ...]\n\n"
                + prompt[-half:]
            )
            print(f"  [VLM] Prompt truncated from {len(prompt):,} to {max_prompt_chars:,} chars")

        content.append({"type": "text", "text": prompt})

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        self._conversation = messages

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.0,
        }

        if tools:
            # Convert tool definitions to OpenAI function calling format
            payload["tools"] = self._format_tools(tools)

        data = self._post(payload)
        return self._parse_response(data)

    def send_tool_result(self, conversation: list[dict],
                         tool_results: list[dict],
                         tools: list[dict] | None = None) -> VLMResponse:
        """Continue conversation with tool results."""
        # Add tool result messages
        for tr in tool_results:
            conversation.append({
                "role": "tool",
                "tool_call_id": tr["tool_use_id"],
                "content": json.dumps(tr["result"], default=str),
            })

        payload = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": 4096,
            "temperature": 0.0,
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        data = self._post(payload)
        return self._parse_response(data)

    def _format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert our tool definitions to OpenAI function calling format."""
        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {})),
                },
            })
        return formatted

    def _post(self, payload: dict) -> dict:
        """Send request with retry on transient failures."""
        retryable = {429, 500, 502, 503}
        for attempt in range(3):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in retryable and attempt < 2:
                    wait = 2 ** attempt * 5
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after and retry_after.isdigit():
                        wait = max(wait, int(retry_after))
                    print(f"  [retry {attempt + 1}/3] HTTP {e.response.status_code}, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                if attempt < 2:
                    print(f"  [retry {attempt + 1}/3] {type(e).__name__}, waiting {5}s...")
                    time.sleep(5)
                else:
                    raise

    def _parse_response(self, data: dict) -> VLMResponse:
        """Parse OpenAI-format response into VLMResponse."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        text = message.get("content", "") or ""
        tool_calls = []

        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}

            tool_calls.append(ToolCall(
                tool_name=func.get("name", ""),
                arguments=args,
                call_id=tc.get("id", ""),
            ))

        # Build conversation state for multi-turn
        assistant_msg = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_msg["tool_calls"] = message.get("tool_calls", [])
        self._conversation.append(assistant_msg)

        usage = data.get("usage", {})
        return VLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            stop_reason=choice.get("finish_reason", ""),
        )

    def get_conversation(self) -> list[dict]:
        """Get the current conversation state."""
        return self._conversation

    @staticmethod
    def _prepare_image(image_path: str, max_size: int = 1200) -> bytes:
        """Resize image if too large to reduce API latency and token usage.

        Args:
            image_path: Path to image file.
            max_size: Max width or height in pixels (default 1200).

        Returns:
            PNG bytes of the (possibly resized) image.
        """
        from PIL import Image
        import io

        img = Image.open(image_path)
        w, h = img.size

        if w > max_size or h > max_size:
            ratio = min(max_size / w, max_size / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Convert to PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
