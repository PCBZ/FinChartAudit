"""T1 Numerical Consistency Agent — detects text-chart value mismatches."""
from __future__ import annotations

import json

from finchartaudit.agents.base import BaseAgent
from finchartaudit.memory.models import AuditFinding, Claim, RiskLevel, Tier
from finchartaudit.prompts.t1_numerical import T1_SYSTEM_PROMPT, build_t1_prompt


class T1NumericalAgent(BaseAgent):
    agent_name = "t1_numerical"
    available_tools = ["traditional_ocr", "rule_check", "query_memory", "html_extract"]

    def execute(self, task: dict) -> list[AuditFinding]:
        image_path = task["image_path"]
        text_context = task.get("text_context", "")
        page = task.get("page", 0)
        chart_id = task.get("chart_id", f"p{page}_c1")

        if not text_context:
            self.memory.audit_trace.log_reasoning(
                self.agent_name, f"No text context for {chart_id}, skipping T1")
            return []

        prompt = build_t1_prompt(text_context=text_context)
        final_text, tool_results = self.run_with_tools(
            image_path=image_path, prompt=prompt, system=T1_SYSTEM_PROMPT)

        return self._parse_response(final_text, chart_id, page)

    def _parse_response(self, text: str, chart_id: str, page: int) -> list[AuditFinding]:
        findings = []
        json_data = self._extract_json(text)
        if not json_data:
            return findings

        for claim_data in json_data.get("claims", []):
            if claim_data.get("match", True) is not False:
                continue

            claim = Claim(
                text=claim_data.get("text_claim", ""),
                page=page,
                metric=claim_data.get("metric", ""),
                value=claim_data.get("text_value"),
                context=claim_data.get("evidence", ""),
            )
            self.memory.add_claim(claim)

            text_val = claim_data.get("text_value", 0)
            chart_val = claim_data.get("chart_value", 0)
            if text_val and chart_val:
                diff_pct = abs(text_val - chart_val) / abs(text_val) * 100
                findings.append(AuditFinding(
                    tier=Tier.T1,
                    category="text_chart",
                    subcategory="value_mismatch",
                    page=page,
                    chart_id=chart_id,
                    risk_level=RiskLevel.HIGH if diff_pct > 10 else RiskLevel.MEDIUM,
                    confidence=min(0.9, diff_pct / 20),
                    description=(
                        f"Text claims {claim_data.get('metric', 'value')} = {text_val}, "
                        f"but chart shows {chart_val} ({diff_pct:.1f}% difference)"
                    ),
                    correction="Verify and correct the discrepancy between text and chart values.",
                    evidence=[claim_data.get("evidence", "")],
                ))

        for f in findings:
            self.memory.add_finding(f)
        return findings

    def _extract_json(self, text: str) -> dict | None:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break
        return None
