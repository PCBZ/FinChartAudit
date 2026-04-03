"""Base agent class with tool-use support."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from finchartaudit.vlm.base import VLMClient, VLMResponse, ToolCall, ToolCallResult
from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.memory.models import AuditFinding
from finchartaudit.tools.registry import get_tool_schemas


class BaseAgent(ABC):
    """Abstract agent that uses a VLM with tool-calling capability."""

    agent_name: str = "base"
    available_tools: list[str] = []

    def __init__(self, vlm: VLMClient, memory: FilingMemory):
        self.vlm = vlm
        self.memory = memory
        self._tool_executors: dict = {}
        self._current_image_path: str = ""  # Tracks the image being analyzed
        self._init_tool_executors()

    def _init_tool_executors(self):
        """Lazy-import and register tool executors."""
        from finchartaudit.tools.rule_check import RuleEngine
        from finchartaudit.tools.query_memory import QueryMemoryTool

        self._rule_engine = RuleEngine()
        self._query_memory = QueryMemoryTool(self.memory)

        self._tool_executors = {
            "rule_check": lambda args: self._rule_engine.run_check(
                args["check_type"], args.get("data", {})),
            "query_memory": lambda args: self._query_memory.run(
                args["query"], args.get("scope", "current_filing")),
        }

        from finchartaudit.tools.html_extract import HtmlFilingExtractor
        self._html_extractor = HtmlFilingExtractor()
        self._tool_executors["html_extract"] = lambda args: self._html_extractor.run(args["file_path"])

    def set_ocr_tool(self, ocr_tool) -> None:
        """Register the OCR tool (must be set after construction if OCR is available)."""
        def _run_ocr(args):
            # Resolve image_id to actual file path
            image_id = args.get("image_id", "")
            image_path = self._resolve_image_path(image_id)
            return ocr_tool.run(image_path, args.get("region", "full"), args.get("mode", "bbox"))
        self._tool_executors["traditional_ocr"] = _run_ocr

    def _resolve_image_path(self, image_id: str) -> str:
        """Resolve an image_id from the VLM to an actual file path.

        The VLM may pass the chart_id, a generic name like 'chart_image',
        or the actual path. We always resolve to self._current_image_path
        since the VLM is analyzing one image at a time.
        """
        from pathlib import Path
        # If it's already a valid file path, use it
        if Path(image_id).exists():
            return image_id
        # Otherwise use the current image being analyzed
        return self._current_image_path

    def set_pdf_tool(self, pdf_tool) -> None:
        """Register the PDF text extraction tool."""
        self._tool_executors["extract_pdf_text"] = lambda args: pdf_tool.run(
            args["page"], args.get("extract_tables", True))

    # ── Abstract methods ──

    @abstractmethod
    def execute(self, task: dict) -> list[AuditFinding]:
        """Execute an audit task. Returns list of findings."""

    def plan(self, task: dict) -> list[dict]:
        """Plan subtasks. Default: single task = [task]."""
        return [task]

    def reflect(self, findings: list[AuditFinding]) -> list[dict]:
        """Reflect on findings and optionally generate follow-up tasks. Default: none."""
        return []

    # ── Tool-use execution loop ──

    def run_with_tools(self, image_path: str, prompt: str,
                       system: str = "") -> tuple[str, list[ToolCallResult]]:
        """Run VLM with tools in a loop until the VLM stops calling tools.

        Returns:
            (final_text, all_tool_results)
        """
        tools = get_tool_schemas(self.available_tools)
        all_results: list[ToolCallResult] = []
        self._current_image_path = image_path  # Track for OCR tool resolution

        # Initial request (sends image)
        response = self.vlm.analyze(image_path, prompt, tools=tools, system=system)

        # Tool-use loop: max 3 rounds to keep latency under 60s
        # Each round may contain multiple parallel tool calls
        max_iterations = 3
        iteration = 0

        while response.tool_calls and iteration < max_iterations:
            iteration += 1

            tool_results_for_api = []

            for tc in response.tool_calls:
                self.memory.audit_trace.log_tool_call(
                    self.agent_name, tc.tool_name,
                    json.dumps(tc.arguments, default=str)[:200])

                result = self._execute_tool(tc.tool_name, tc.arguments)

                # Truncate large OCR results to reduce token usage in follow-up calls
                result_compact = self._compact_result(result)

                result_str = json.dumps(result_compact, default=str)[:300]
                self.memory.audit_trace.log_tool_result(
                    self.agent_name, tc.tool_name, result_str)

                all_results.append(ToolCallResult(
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                    result=result,
                    call_id=tc.call_id,
                ))

                tool_results_for_api.append({
                    "tool_use_id": tc.call_id,
                    "result": result_compact,
                })

            # Send results back — conversation history managed by VLM client
            conversation = getattr(self.vlm, 'get_conversation', lambda: [])()
            response = self.vlm.send_tool_result(
                conversation, tool_results_for_api, tools=tools)

        # If loop exhausted (VLM still wants to call tools), force a final answer
        if response.tool_calls and iteration >= max_iterations:
            conversation = getattr(self.vlm, 'get_conversation', lambda: [])()
            # Send a nudge to produce the final JSON
            force_msg = [{
                "tool_use_id": tc.call_id,
                "result": {"note": "Tool call limit reached. Please provide your final JSON assessment now based on the information you already have."},
            } for tc in response.tool_calls]
            response = self.vlm.send_tool_result(conversation, force_msg, tools=None)

        if response.text:
            self.memory.audit_trace.log_reasoning(self.agent_name, response.text[:300])

        return response.text, all_results

    @staticmethod
    def _compact_result(result: dict) -> dict:
        """Compact tool results to reduce token usage in multi-turn conversations."""
        if "text_blocks" in result:
            blocks = result["text_blocks"]
            if len(blocks) > 15:
                # Keep only top 15 blocks by confidence, summarize the rest
                sorted_blocks = sorted(blocks, key=lambda b: b.get("confidence", 0), reverse=True)
                result = {
                    "text_blocks": sorted_blocks[:15],
                    "total_blocks": len(blocks),
                    "truncated": True,
                }
        return result

    def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch tool execution."""
        executor = self._tool_executors.get(tool_name)
        if executor is None:
            return {"error": f"Tool '{tool_name}' not available for this agent."}
        try:
            return executor(arguments)
        except Exception as e:
            return {"error": str(e)}
