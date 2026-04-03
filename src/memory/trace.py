"""Audit trace logger — records every agent action for provenance."""
from __future__ import annotations

from .models import TraceEntry


class AuditTracer:
    """Manages a sequential log of agent actions."""

    def __init__(self):
        self._entries: list[TraceEntry] = []

    def log_reasoning(self, agent: str, thought: str) -> None:
        self._entries.append(TraceEntry(
            agent=agent, action="vlm_reasoning",
            input_summary="", output_summary=thought,
        ))

    def log_tool_call(self, agent: str, tool_name: str, input_summary: str) -> None:
        self._entries.append(TraceEntry(
            agent=agent, action="tool_call",
            tool_name=tool_name, input_summary=input_summary,
        ))

    def log_tool_result(self, agent: str, tool_name: str, output_summary: str) -> None:
        self._entries.append(TraceEntry(
            agent=agent, action="tool_result",
            tool_name=tool_name, output_summary=output_summary,
        ))

    def log_finding(self, agent: str, finding_summary: str) -> None:
        self._entries.append(TraceEntry(
            agent=agent, action="finding", output_summary=finding_summary,
        ))

    def log_decision(self, agent: str, decision: str) -> None:
        self._entries.append(TraceEntry(
            agent=agent, action="decision", decision=decision,
        ))

    def get_trace(self) -> list[TraceEntry]:
        return list(self._entries)

    def get_trace_for_agent(self, agent_name: str) -> list[TraceEntry]:
        return [e for e in self._entries if e.agent == agent_name]

    def export_json(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    def print_trace(self) -> None:
        for i, e in enumerate(self._entries, 1):
            tool_info = f" [{e.tool_name}]" if e.tool_name else ""
            content = e.output_summary or e.input_summary or e.decision or ""
            print(f"[{i}] {e.agent} | {e.action}{tool_info}: {content[:120]}")
