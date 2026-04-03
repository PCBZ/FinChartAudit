"""T2 Visual Encoding Auditor — detects 12 types of misleading charts."""
from __future__ import annotations

import json

from finchartaudit.agents.base import BaseAgent
from finchartaudit.memory.models import AuditFinding, ChartRecord, RiskLevel, Tier
from finchartaudit.prompts.t2_visual import T2_SYSTEM_PROMPT, build_detection_prompt, SEC_RULE_MAPPING


class T2VisualAgent(BaseAgent):
    """Detects misleading visual encoding in charts using VLM + OCR + rules."""

    agent_name = "t2_visual"
    available_tools = ["traditional_ocr", "rule_check", "query_memory"]

    def execute(self, task: dict) -> list[AuditFinding]:
        """Analyze a chart image for misleading techniques.

        Args:
            task: {"image_path": str, "page": int, "chart_id": str}
        """
        image_path = task["image_path"]
        page = task.get("page", 0)
        chart_id = task.get("chart_id", f"p{page}_c1")

        prompt = build_detection_prompt(chart_id=chart_id, page=page)

        # Run VLM with tool-use loop
        final_text, tool_results = self.run_with_tools(
            image_path=image_path,
            prompt=prompt,
            system=T2_SYSTEM_PROMPT,
        )

        # Parse VLM response into findings
        findings = self._parse_response(final_text, chart_id, page, tool_results)

        # Register chart in memory
        chart_record = self._build_chart_record(final_text, chart_id, page, image_path)
        if chart_record:
            self.memory.register_chart(chart_record)

        return findings

    def _parse_response(self, text: str, chart_id: str, page: int,
                        tool_results: list) -> list[AuditFinding]:
        """Parse VLM's JSON response into AuditFinding objects."""
        findings = []

        # Try to extract JSON from the response
        json_data = self._extract_json(text)
        if not json_data or ("misleaders" not in json_data and "completeness" not in json_data):
            self.memory.audit_trace.log_reasoning(
                self.agent_name, f"Non-JSON response for {chart_id}: {text[:200]}")
            return findings

        tool_names = [tr.tool_name for tr in tool_results]

        # Part A: Misleader findings
        for misleader_name, assessment in json_data.get("misleaders", {}).items():
            if not isinstance(assessment, dict) or not assessment.get("present", False):
                continue
            confidence = float(assessment.get("confidence", 0.5))
            if confidence < 0.3:
                continue

            sec_rule = SEC_RULE_MAPPING.get(misleader_name, "")
            evidence_list = [assessment.get("evidence", "")]
            if sec_rule:
                evidence_list.append(f"SEC basis: {sec_rule}")

            findings.append(AuditFinding(
                tier=Tier.T2,
                category="misleader",
                subcategory=misleader_name,
                page=page,
                chart_id=chart_id,
                risk_level=self._assess_risk(misleader_name, confidence),
                confidence=confidence,
                description=assessment.get("evidence", f"{misleader_name} detected"),
                correction=self._suggest_correction(misleader_name),
                evidence=evidence_list,
                tool_calls=tool_names,
            ))

        # Part B: Completeness findings
        for check_name, assessment in json_data.get("completeness", {}).items():
            if not isinstance(assessment, dict) or not assessment.get("present", False):
                continue
            confidence = float(assessment.get("confidence", 0.5))
            if confidence < 0.3:
                continue

            sec_rule = SEC_RULE_MAPPING.get(check_name, "")
            evidence_list = [assessment.get("evidence", "")]
            if sec_rule:
                evidence_list.append(f"SEC basis: {sec_rule}")

            findings.append(AuditFinding(
                tier=Tier.T2,
                category="completeness",
                subcategory=check_name,
                page=page,
                chart_id=chart_id,
                risk_level=self._assess_completeness_risk(check_name, confidence),
                confidence=confidence,
                description=assessment.get("evidence", f"{check_name}"),
                correction=self._suggest_completeness_fix(check_name),
                evidence=evidence_list,
                tool_calls=tool_names,
            ))

        for f in findings:
            self.memory.add_finding(f)
            self.memory.audit_trace.log_finding(
                self.agent_name, f"{f.subcategory} ({f.risk_level}, {f.confidence:.0%})")

        return findings

    def reflect(self, findings: list[AuditFinding]) -> list[dict]:
        """Check if important checks were missed and suggest follow-ups."""
        follow_ups = []
        found_types = {f.subcategory for f in findings}

        has_axis_check = "truncated axis" in found_types
        if not has_axis_check:
            trace = self.memory.audit_trace.get_trace()
            for entry in trace:
                if entry.tool_name == "traditional_ocr" and entry.output_summary:
                    if any(c.isdigit() for c in entry.output_summary):
                        follow_ups.append({
                            "reason": "OCR found numeric values but truncated_axis not checked",
                            "action": "run_rule_check",
                            "check_type": "truncated_axis",
                        })
                        break

        return follow_ups

    def _build_chart_record(self, text: str, chart_id: str,
                            page: int, image_path: str) -> ChartRecord | None:
        """Extract chart metadata from VLM response."""
        json_data = self._extract_json(text)
        if not json_data:
            return ChartRecord(chart_id=chart_id, page=page, image_path=image_path)

        return ChartRecord(
            chart_id=chart_id,
            page=page,
            metric_name=json_data.get("metric_name", ""),
            is_gaap=json_data.get("is_gaap", True),
            chart_type=json_data.get("chart_type", ""),
            time_window_start=json_data.get("time_window_start", ""),
            time_window_end=json_data.get("time_window_end", ""),
            image_path=image_path,
        )

    def _extract_json(self, text: str) -> dict | None:
        """Try to extract JSON from VLM response text."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON block in markdown
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try to find { ... } block
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

    def _assess_risk(self, misleader_name: str, confidence: float) -> str:
        """Assign risk level based on misleader type and confidence."""
        high_impact = {"truncated axis", "misrepresentation", "inverted axis", "3d"}
        medium_impact = {"dual axis", "inappropriate axis range", "inconsistent tick intervals"}

        if misleader_name in high_impact and confidence > 0.7:
            return RiskLevel.HIGH
        if misleader_name in high_impact or confidence > 0.8:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _suggest_correction(self, misleader_name: str) -> str:
        """Generate a correction suggestion for a misleader type."""
        corrections = {
            "truncated axis": "Set Y-axis origin to 0 to avoid exaggerating visual differences.",
            "inverted axis": "Ensure axis values increase in the expected direction.",
            "misrepresentation": "Ensure visual encodings (height, area, angle) accurately represent data values.",
            "3d": "Use 2D chart instead of 3D to avoid perspective distortion.",
            "dual axis": "Consider separate charts or normalize scales to avoid misleading correlations.",
            "inappropriate use of pie chart": "Use a bar chart for better comparison of values.",
            "inappropriate use of line chart": "Use a bar chart for categorical data instead of a line chart.",
            "inconsistent binning size": "Use equal-width bins for accurate distribution representation.",
            "inconsistent tick intervals": "Use evenly spaced tick marks on all axes.",
            "discretized continuous variable": "Preserve continuous data representation or explain discretization.",
            "inappropriate item order": "Order items logically (alphabetically, by value, or chronologically).",
            "inappropriate axis range": "Choose an axis range that fairly represents the data variation.",
        }
        return corrections.get(misleader_name, "Review chart for potential misleading elements.")

    def _assess_completeness_risk(self, check_name: str, confidence: float) -> str:
        """Assign risk level for completeness issues."""
        high_impact = {
            "missing_x_axis_values", "missing_y_axis_values",
            "missing_nongaap_label", "missing_reconciliation_ref",
        }
        medium_impact = {
            "missing_x_axis_label", "missing_y_axis_label",
            "missing_legend", "missing_data_units", "missing_chart_title",
        }

        if check_name in high_impact and confidence > 0.7:
            return RiskLevel.HIGH
        if check_name in high_impact or check_name in medium_impact:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _suggest_completeness_fix(self, check_name: str) -> str:
        """Generate fix suggestion for completeness issues."""
        fixes = {
            "missing_chart_title": "Add a descriptive title to the chart.",
            "missing_y_axis_label": "Add a label describing the Y-axis (e.g., 'Revenue ($M)').",
            "missing_x_axis_label": "Add a label or tick labels to the X-axis.",
            "missing_y_axis_values": "Add numeric tick values to the Y-axis.",
            "missing_x_axis_values": "Add tick labels (years, categories) to the X-axis.",
            "missing_legend": "Add a legend identifying each data series.",
            "missing_data_units": "Specify units ($, %, millions, etc.) on axes or in title.",
            "missing_data_source": "Add 'Source: ...' attribution below the chart.",
            "missing_nongaap_label": "Clearly label Non-GAAP/Adjusted metrics as such.",
            "missing_reconciliation_ref": "Add reference to GAAP reconciliation (e.g., 'See Appendix A').",
            "missing_base_period": "Specify the base period and value for indexed comparisons.",
        }
        return fixes.get(check_name, "Add missing information to the chart.")
