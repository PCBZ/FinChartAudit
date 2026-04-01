"""T2 Pipeline Agent — pre-compute OCR + rules, single VLM call.

Unlike the agentic T2 (multi-turn tool-use), this agent:
1. Always runs OCR first (deterministic)
2. Always runs applicable rule checks (deterministic)
3. Injects evidence into a single VLM prompt
4. Makes ONE VLM call for final judgment

This avoids multi-turn reasoning degradation in small models (e.g., Haiku)
while still benefiting from OCR + rule evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

from finchartaudit.memory.models import AuditFinding, ChartRecord, RiskLevel, Tier
from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.vlm.base import VLMClient
from finchartaudit.tools.rule_check import RuleEngine
from finchartaudit.prompts.t2_visual import MISLEADER_DEFINITIONS, COMPLETENESS_CHECKS, SEC_RULE_MAPPING


PIPELINE_SYSTEM_PROMPT = """You are a financial chart auditor. You detect TWO categories of issues:

PART A — Misleading visual encoding (12 Misviz types): Does the chart visually distort the data?
PART B — Completeness issues (11 types): Is required information missing from the chart?

You have been given PRE-EXTRACTED evidence from OCR and rule checks.
Use this evidence together with your own visual analysis to make your final judgment.
Be precise — only flag issues you have evidence for."""


PIPELINE_PROMPT = """Analyze this chart image (chart_id: {chart_id}, page: {page}).

=== PRE-EXTRACTED EVIDENCE ===

OCR Text (full image):
{ocr_text}

OCR Axis Values (Y-axis region):
{ocr_axis}

Rule Check Results:
{rule_results}

=== PART A: Misleading Visual Encoding ===
Check each of these 12 misleader types:
{misleader_list}

=== PART B: Completeness Issues ===
Check each of these 11 completeness items:
{completeness_list}

Based on the image AND the pre-extracted evidence, respond with this JSON:
{{
  "chart_type": "...",
  "metric_name": "...",
  "is_gaap": true/false,
  "misleaders": {{
    "truncated axis": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    ...
  }},
  "completeness": {{
    "missing_chart_title": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    ...
  }}
}}"""


class T2PipelineAgent:
    """Single-turn pipeline: OCR → Rules → VLM judgment."""

    def __init__(self, vlm: VLMClient, memory: FilingMemory):
        self.vlm = vlm
        self.memory = memory
        self._ocr_tool = None
        self._rule_engine = RuleEngine()

    def set_ocr_tool(self, ocr_tool):
        self._ocr_tool = ocr_tool

    def execute(self, task: dict) -> list[AuditFinding]:
        image_path = task["image_path"]
        page = task.get("page", 0)
        chart_id = task.get("chart_id", f"p{page}_c1")

        # Step 1: Run OCR (deterministic, no VLM involved)
        ocr_text = ""
        ocr_axis = ""
        axis_values = []
        right_axis_values = []
        x_axis_values = []

        if self._ocr_tool and Path(image_path).exists():
            try:
                full_result = self._ocr_tool.run(image_path, "full", "bbox")
                ocr_text = self._format_ocr(full_result)

                y_result = self._ocr_tool.run(image_path, "y_axis", "text")
                ocr_axis = self._format_ocr(y_result)
                axis_values = self._extract_numbers(y_result)

                # Right Y-axis for dual axis detection
                right_result = self._ocr_tool.run(image_path, "right_axis", "text")
                right_axis_values = self._extract_numbers(right_result)

                # X-axis for binning detection
                x_result = self._ocr_tool.run(image_path, "x_axis", "text")
                x_axis_values = self._extract_numbers(x_result)
            except Exception as e:
                ocr_text = f"OCR failed: {e}"

            self.memory.audit_trace.log_tool_call("t2_pipeline", "traditional_ocr", "full+y_axis+right+x")
            self.memory.audit_trace.log_tool_result("t2_pipeline", "traditional_ocr", ocr_text[:200])

        # Step 2: Run rule checks (deterministic)
        rule_results = []
        if axis_values:
            # Truncated axis & broken scale
            for check_type in ["truncated_axis", "broken_scale"]:
                try:
                    result = self._rule_engine.run_check(check_type, {
                        "axis_values": axis_values,
                        "chart_type": "bar",
                    })
                    rule_results.append(f"{check_type}: {result['explanation']}")
                    self.memory.audit_trace.log_tool_call("t2_pipeline", "rule_check", check_type)
                except Exception:
                    pass

            # Inverted axis
            try:
                result = self._rule_engine.run_check("inverted_axis", {
                    "axis_values": axis_values,
                })
                if result["is_inverted"]:
                    rule_results.append(f"inverted_axis: {result['explanation']}")
                self.memory.audit_trace.log_tool_call("t2_pipeline", "rule_check", "inverted_axis")
            except Exception:
                pass

            # Inappropriate axis range
            try:
                result = self._rule_engine.run_check("inappropriate_axis_range", {
                    "axis_values": axis_values,
                })
                if result["is_inappropriate"]:
                    rule_results.append(f"inappropriate_axis_range: {result['explanation']}")
                self.memory.audit_trace.log_tool_call("t2_pipeline", "rule_check", "inappropriate_axis_range")
            except Exception:
                pass

        # Dual axis (needs both left and right)
        if axis_values and right_axis_values:
            try:
                result = self._rule_engine.run_check("dual_axis", {
                    "left_axis_values": axis_values,
                    "right_axis_values": right_axis_values,
                })
                if result["has_dual_axis"]:
                    rule_results.append(f"dual_axis: {result['explanation']}")
                self.memory.audit_trace.log_tool_call("t2_pipeline", "rule_check", "dual_axis")
            except Exception:
                pass

        # Inconsistent binning (from X-axis)
        if len(x_axis_values) >= 3:
            try:
                result = self._rule_engine.run_check("inconsistent_binning", {
                    "bin_edges": x_axis_values,
                })
                if result["is_inconsistent"]:
                    rule_results.append(f"inconsistent_binning: {result['explanation']}")
                self.memory.audit_trace.log_tool_call("t2_pipeline", "rule_check", "inconsistent_binning")
            except Exception:
                pass

        # Step 3: Build prompt with evidence
        misleader_list = "\n".join(f"- {k}: {v}" for k, v in MISLEADER_DEFINITIONS.items())
        completeness_list = "\n".join(f"- {k}: {v}" for k, v in COMPLETENESS_CHECKS.items())

        prompt = PIPELINE_PROMPT.format(
            chart_id=chart_id,
            page=page,
            ocr_text=ocr_text or "No OCR results available.",
            ocr_axis=ocr_axis or "No axis values extracted.",
            rule_results="\n".join(rule_results) if rule_results else "No rule checks applicable (no numeric axis values found).",
            misleader_list=misleader_list,
            completeness_list=completeness_list,
        )

        # Step 4: Single VLM call
        response = self.vlm.analyze(
            image_path, prompt, tools=None, system=PIPELINE_SYSTEM_PROMPT)

        self.memory.audit_trace.log_reasoning("t2_pipeline", response.text[:300])

        # Step 5: Parse response
        findings = self._parse_response(response.text, chart_id, page)

        # Register chart
        chart_record = ChartRecord(chart_id=chart_id, page=page, image_path=image_path)
        self.memory.register_chart(chart_record)

        return findings

    def _parse_response(self, text: str, chart_id: str, page: int) -> list[AuditFinding]:
        findings = []
        json_data = self._extract_json(text)
        if not json_data:
            return findings

        # Misleaders
        for name, assessment in json_data.get("misleaders", {}).items():
            if not isinstance(assessment, dict) or not assessment.get("present", False):
                continue
            confidence = float(assessment.get("confidence", 0.5))
            if confidence < 0.3:
                continue

            sec_rule = SEC_RULE_MAPPING.get(name, "")
            evidence = [assessment.get("evidence", "")]
            if sec_rule:
                evidence.append(f"SEC basis: {sec_rule}")

            findings.append(AuditFinding(
                tier=Tier.T2,
                category="misleader",
                subcategory=name,
                page=page,
                chart_id=chart_id,
                risk_level=self._assess_risk(name, confidence),
                confidence=confidence,
                description=assessment.get("evidence", f"{name} detected"),
                correction=self._suggest_correction(name),
                evidence=evidence,
                tool_calls=["traditional_ocr", "rule_check"],
            ))

        # Completeness
        for name, assessment in json_data.get("completeness", {}).items():
            if not isinstance(assessment, dict) or not assessment.get("present", False):
                continue
            confidence = float(assessment.get("confidence", 0.5))
            if confidence < 0.3:
                continue

            findings.append(AuditFinding(
                tier=Tier.T2,
                category="completeness",
                subcategory=name,
                page=page,
                chart_id=chart_id,
                risk_level=RiskLevel.MEDIUM if confidence > 0.7 else RiskLevel.LOW,
                confidence=confidence,
                description=assessment.get("evidence", name),
                correction=f"Add missing {name.replace('missing_', '').replace('_', ' ')}.",
                evidence=[assessment.get("evidence", "")],
                tool_calls=["traditional_ocr"],
            ))

        for f in findings:
            self.memory.add_finding(f)

        return findings

    def _format_ocr(self, result: dict) -> str:
        blocks = result.get("text_blocks", [])
        if not blocks:
            return "No text detected."
        lines = []
        for b in blocks[:20]:  # Limit to reduce prompt size
            text = b.get("text", "")
            conf = b.get("confidence", 0)
            if conf > 0.5 and text.strip():
                lines.append(text.strip())
        return "\n".join(lines) if lines else "No confident text detected."

    def _extract_numbers(self, result: dict) -> list[float]:
        import re
        numbers = []
        for b in result.get("text_blocks", []):
            text = b.get("text", "")
            for match in re.findall(r'-?\d+\.?\d*', text):
                try:
                    numbers.append(float(match))
                except ValueError:
                    pass
        return sorted(set(numbers))

    def _assess_risk(self, name: str, confidence: float) -> str:
        high_impact = {"truncated axis", "misrepresentation", "inverted axis", "3d"}
        if name in high_impact and confidence > 0.7:
            return RiskLevel.HIGH
        if name in high_impact or confidence > 0.8:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _suggest_correction(self, name: str) -> str:
        corrections = {
            "truncated axis": "Set Y-axis origin to 0.",
            "inverted axis": "Ensure axis values increase in expected direction.",
            "misrepresentation": "Ensure visual encodings accurately represent data.",
            "3d": "Use 2D chart instead of 3D.",
            "dual axis": "Consider separate charts or normalize scales.",
            "inappropriate use of pie chart": "Use a bar chart for comparison.",
            "inappropriate use of line chart": "Use a bar chart for categorical data.",
            "inconsistent binning size": "Use equal-width bins.",
            "inconsistent tick intervals": "Use evenly spaced tick marks.",
            "discretized continuous variable": "Preserve continuous representation.",
            "inappropriate item order": "Order items logically.",
            "inappropriate axis range": "Choose a fair axis range.",
        }
        return corrections.get(name, "Review chart for potential issues.")

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
