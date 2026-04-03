"""Abstract VLM client interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict
    call_id: str = ""


@dataclass
class VLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    stop_reason: str = ""


@dataclass
class ToolCallResult:
    tool_name: str
    arguments: dict
    result: dict
    call_id: str = ""


class VLMClient(ABC):
    """Abstract base for vision-language model clients."""

    @abstractmethod
    def analyze(self, image_path: str, prompt: str,
                tools: list[dict] | None = None,
                system: str = "") -> VLMResponse:
        """Send image + prompt to VLM. Returns response with optional tool calls."""

    @abstractmethod
    def send_tool_result(self, conversation: list[dict],
                         tool_results: list[dict],
                         tools: list[dict] | None = None) -> VLMResponse:
        """Continue a conversation by sending tool results back to the VLM."""
