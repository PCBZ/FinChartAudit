"""Qwen2.5-VL client — stub for B to implement via OpenAI-compatible API."""
from __future__ import annotations

from .base import VLMClient, VLMResponse


class QwenClient(VLMClient):
    """Qwen2.5-VL via OpenAI-compatible endpoint (vLLM / TGI)."""

    def __init__(self, endpoint: str = "http://localhost:8000/v1"):
        self.endpoint = endpoint
        # TODO: B to implement — Qwen serves via vLLM with OpenAI-compatible API

    def analyze(self, image_path: str, prompt: str,
                tools: list[dict] | None = None,
                system: str = "") -> VLMResponse:
        # TODO: Implement using httpx or openai client
        raise NotImplementedError("QwenClient not yet implemented. Assign to Member B.")

    def send_tool_result(self, conversation: list[dict],
                         tool_results: list[dict],
                         tools: list[dict] | None = None) -> VLMResponse:
        raise NotImplementedError("QwenClient not yet implemented. Assign to Member B.")
