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


PIPELINE_SYSTEM_PROMPT = """You are a financial chart auditor detecting misleading visual encodings and completeness issues.

RULES FOR SECTION A (structural issues with OCR data):
- Trust [CLEAN] verdicts — do NOT override them unless you see a clear discrepancy in the image.
- Trust [FLAGGED] verdicts — include them.
- For [INFO] items — make your OWN judgment based on the image. The rule engine is uncertain.
- Do NOT confuse: "truncated axis" (Y starts above 0 in bar chart) vs "inappropriate axis range" (range too narrow).
- Line/scatter charts are ALLOWED to have non-zero Y-axis origins.

RULES FOR SECTION B (visual issues — your eyes only):
- Look carefully at the chart. Flag issues you can see.
- "misrepresentation" = bar heights/pie angles don't match the data labels.
- "3d" = 3D perspective distorts perception of values.
- Be thorough — these issues can only be detected visually."""


PIPELINE_PROMPT = """Analyze this chart image (chart_id: {chart_id}, page: {page}).

=== SECTION A: STRUCTURAL ISSUES (use OCR evidence) ===

OCR extracted Y-axis values: {ocr_axis}
OCR extracted X-axis values: {ocr_x_axis}
Rule engine verdicts:
{rule_verdicts}

Based on the OCR data AND the image, check these structural issues:
- truncated axis: Y-axis does not start from 0 in a bar/area chart, exaggerating differences. (Line charts are exempt.)
- inverted axis: Axis values run in reverse order (e.g., decreasing from bottom to top).
- inappropriate axis range: Y-axis range is extremely narrow relative to values, exaggerating tiny differences.
- dual axis: Two Y-axes with different scales suggesting false correlations.
- inconsistent tick intervals: Tick marks not evenly spaced.
- discretized continuous variable: Continuous data forced into discrete categories.
- inconsistent binning size: Histogram bins with unequal widths.

IMPORTANT: If the rule engine says "not truncated" or "axis is normal", you should AGREE unless
you can point to a SPECIFIC discrepancy between the OCR values and what you see in the image.

=== SECTION B: VISUAL ISSUES (use your eyes only — be thorough) ===

Carefully examine the chart image for these visual issues. Do NOT reference the OCR data above.
For each, look at the actual visual rendering:
- misrepresentation: Do bar heights, pie slice angles, or area sizes MATCH the data labels shown?
  Compare specific values: if a label says "50" and another says "100", is the second bar exactly 2x taller?
- 3d: Is this a 3D chart? Does the perspective make front bars/slices appear larger than back ones?
- inappropriate use of pie chart: Is this a pie chart showing values that don't sum to a whole, or with too many slices?
- inappropriate use of line chart: Is this a line chart connecting categorical (non-sequential) data points?
- inappropriate item order: Are items arranged to create a false visual trend (e.g., ascending order suggesting growth)?

=== SECTION C: COMPLETENESS ===
{completeness_list}

Respond with this JSON:
{{
  "chart_type": "...",
  "metric_name": "...",
  "is_gaap": true/false,
  "misleaders": {{
    "truncated axis": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "inverted axis": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "misrepresentation": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "3d": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "dual axis": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "inappropriate use of pie chart": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "inappropriate use of line chart": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "inconsistent binning size": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "inconsistent tick intervals": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "discretized continuous variable": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "inappropriate item order": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "inappropriate axis range": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}}
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

        # Step 3: Build structured rule verdicts (not raw strings)
        rule_verdicts = self._build_rule_verdicts(
            axis_values, right_axis_values, x_axis_values, rule_results)

        completeness_list = "\n".join(f"- {k}: {v}" for k, v in COMPLETENESS_CHECKS.items())

        # Format X-axis values
        ocr_x_str = ", ".join(str(v) for v in x_axis_values[:15]) if x_axis_values else "Not extracted"

        prompt = PIPELINE_PROMPT.format(
            chart_id=chart_id,
            page=page,
            ocr_axis=ocr_axis or "No Y-axis values extracted.",
            ocr_x_axis=ocr_x_str,
            rule_verdicts=rule_verdicts,
            completeness_list=completeness_list,
        )

        # Step 4: Single VLM call
        response = self.vlm.analyze(
            image_path, prompt, tools=None, system=PIPELINE_SYSTEM_PROMPT)

        self.memory.audit_trace.log_reasoning("t2_pipeline", response.text[:300])

        # Step 5: Parse response
        findings = self._parse_response(response.text, chart_id, page)

        # Step 6: Post-processing — rule veto for structural types
        findings = self._apply_rule_veto(findings, axis_values, rule_results)

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

    # Rules where we trust the verdict enough to label [CLEAN]/[FLAGGED]
    # (validated by routing simulation: veto improves F1)
    RELIABLE_RULES = {"truncated_axis", "dual_axis"}

    def _build_rule_verdicts(self, axis_values: list, right_axis_values: list,
                              x_axis_values: list, raw_rule_results: list[str]) -> str:
        """Build structured rule verdicts that guide the VLM.

        Only RELIABLE rules get [CLEAN]/[FLAGGED] labels.
        Unreliable rules get neutral [INFO] labels — VLM decides on its own.
        """
        lines = []
        has_y = len(axis_values) > 0

        if not has_y:
            lines.append("No numeric Y-axis values extracted by OCR. Rule checks could not run.")
            lines.append("Use your visual analysis only.")
            return "\n".join(lines)

        # === RELIABLE: truncated_axis — rule verdict is authoritative ===
        trunc_flagged = any("instead of 0" in r.lower() or "exaggerated" in r.lower()
                           for r in raw_rule_results if r.startswith("truncated_axis:"))
        if trunc_flagged:
            lines.append(f"[FLAGGED] truncated_axis: Y-axis starts at {min(axis_values)}, not 0.")
        elif axis_values and min(axis_values) <= 0:
            lines.append(f"[CLEAN] truncated_axis: Y-axis includes 0 (min={min(axis_values)}). NOT truncated.")
        else:
            lines.append(f"[CLEAN] truncated_axis: Y-axis min={min(axis_values)}. Rule did not flag.")

        # === RELIABLE: dual_axis — presence of right Y-axis is objective ===
        dual_flagged = any(r.startswith("dual_axis:") for r in raw_rule_results)
        if dual_flagged:
            lines.append(f"[FLAGGED] dual_axis: Left and right Y-axes detected with different scales.")
        elif right_axis_values:
            lines.append(f"[INFO] dual_axis: Right Y-axis values found: {right_axis_values[:6]}. Verify visually.")
        else:
            lines.append(f"[CLEAN] dual_axis: No right Y-axis detected by OCR.")

        # === UNRELIABLE: inverted_axis — OCR read order ≠ axis direction ===
        # Rule has high FP (28/32 rule-caused). Give raw data, let VLM judge.
        inv_flagged = any(r.startswith("inverted_axis:") for r in raw_rule_results)
        if inv_flagged:
            lines.append(f"[INFO] inverted_axis: OCR reads values top-to-bottom as increasing "
                        f"({axis_values[:4]}...). This MAY indicate inverted axis, or normal read order. "
                        f"Check the image: do smaller values appear at the TOP of the Y-axis?")
        else:
            lines.append(f"[INFO] inverted_axis: Y-axis values {axis_values[:6]}. "
                        f"Use image to determine if axis direction is correct.")

        # === UNRELIABLE: inappropriate_axis_range — rule rarely fires correctly ===
        iar_flagged = any(r.startswith("inappropriate_axis_range:") for r in raw_rule_results)
        val_range = max(axis_values) - min(axis_values) if axis_values else 0
        if iar_flagged:
            lines.append(f"[INFO] inappropriate_axis_range: Range {min(axis_values)}-{max(axis_values)} "
                        f"(span={val_range:.1f}) flagged as narrow. Verify: is this a bar/area chart?")
        else:
            lines.append(f"[INFO] inappropriate_axis_range: Range {min(axis_values)}-{max(axis_values)} "
                        f"(span={val_range:.1f}). Use image to judge if range exaggerates differences.")

        # === UNRELIABLE: inconsistent tick intervals — broken_scale rule noisy ===
        broken_flagged = any("inconsistent" in r.lower() and r.startswith("broken_scale:")
                            for r in raw_rule_results)
        if broken_flagged:
            lines.append(f"[INFO] inconsistent_tick_intervals: Rule detected uneven spacing in "
                        f"values {axis_values[:8]}. Verify visually.")
        else:
            lines.append(f"[INFO] inconsistent_tick_intervals: Values {axis_values[:8]}. "
                        f"Check image for even tick spacing.")

        # Inconsistent binning
        bin_flagged = any(r.startswith("inconsistent_binning:") for r in raw_rule_results)
        if bin_flagged:
            lines.append(f"[INFO] inconsistent_binning: X-axis bin widths appear unequal.")

        return "\n".join(lines)

    def _apply_rule_veto(self, findings: list[AuditFinding],
                         axis_values: list, rule_results: list[str]) -> list[AuditFinding]:
        """Post-processing: veto VLM findings that contradict rule verdicts.

        Strategy F from routing simulation:
        - truncated_axis: require rule confirmation (F1: 35.7% → 69.0%)
        - dual_axis: veto if rule says no dual axis detected
        """
        if not axis_values:
            return findings  # No OCR → can't veto

        # Parse which rules flagged
        trunc_flagged = any("instead of 0" in r.lower() or "exaggerated" in r.lower()
                           for r in rule_results if r.startswith("truncated_axis:"))
        dual_flagged = any(r.startswith("dual_axis:") for r in rule_results)

        vetoed = []
        for f in findings:
            if f.category != "misleader":
                vetoed.append(f)
                continue

            name = f.subcategory
            # Truncated axis: strict — must have rule confirmation
            if name == "truncated axis" and not trunc_flagged:
                self.memory.audit_trace.log_decision(
                    "t2_pipeline", f"VETO truncated_axis: rule says clean, VLM confidence={f.confidence}")
                continue

            # Dual axis: veto if rule didn't detect right Y-axis
            if name == "dual axis" and not dual_flagged:
                self.memory.audit_trace.log_decision(
                    "t2_pipeline", f"VETO dual_axis: no right Y-axis in OCR")
                continue

            vetoed.append(f)

        return vetoed

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
