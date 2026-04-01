"""Prompts for T2 Visual Encoding Auditor."""

MISLEADER_DEFINITIONS = {
    "truncated axis": "Y-axis does not start from 0, exaggerating visual differences between values.",
    "inverted axis": "Axis values are in reverse order (e.g., decreasing from bottom to top).",
    "misrepresentation": "Visual encoding (bar height, area, angle) does not accurately represent the data values.",
    "3d": "3D rendering distorts the visual perception of data values through perspective effects.",
    "dual axis": "Two y-axes with different scales that can suggest misleading correlations.",
    "inappropriate use of pie chart": "Pie chart used where unsuitable (values don't sum to whole, too many slices, etc.).",
    "inappropriate use of line chart": "Line chart used for categorical or non-sequential data, implying false continuity.",
    "inconsistent binning size": "Histogram bins have unequal widths, distorting the perceived distribution.",
    "inconsistent tick intervals": "Tick marks on an axis are not evenly spaced, distorting data perception.",
    "discretized continuous variable": "Continuous data forced into discrete categories, losing granularity.",
    "inappropriate item order": "Items ordered in a misleading way that suggests a trend where none exists.",
    "inappropriate axis range": "Axis range chosen to exaggerate or minimize apparent variation in the data.",
}

COMPLETENESS_CHECKS = {
    "missing_chart_title": "Chart has no title or heading describing what it shows.",
    "missing_y_axis_label": "Y-axis has no name/description (e.g., no 'Revenue ($M)' label).",
    "missing_x_axis_label": "X-axis has no name/description or tick labels (e.g., no year labels, no category names).",
    "missing_y_axis_values": "Y-axis has no numeric tick values — reader cannot determine scale.",
    "missing_x_axis_values": "X-axis has no tick labels — reader cannot identify data points or time periods.",
    "missing_legend": "Chart shows multiple data series (multiple lines, bar groups, colors) but has no legend identifying them.",
    "missing_data_units": "No units specified anywhere (no $, %, millions, etc.) — reader cannot interpret values.",
    "missing_data_source": "No data source attribution (no 'Source: ...' text).",
    "missing_nongaap_label": "Chart shows a Non-GAAP/Adjusted metric but does not label it as Non-GAAP.",
    "missing_reconciliation_ref": "Chart shows a Non-GAAP metric but has no reference to where the GAAP reconciliation can be found.",
    "missing_base_period": "Index or comparison chart (e.g., TSR) does not state the base period or base value.",
}

# SEC rule mapping — associates each check with its regulatory basis
SEC_RULE_MAPPING = {
    # Misleader types
    "truncated axis": "CPA Journal best practices; C&DI 102.10 (misleading presentation)",
    "misrepresentation": "Reg S-K Item 10(e)(1)(i)(A) (fair presentation)",
    "3d": "CPA Journal best practices (visual distortion)",
    "inverted axis": "CPA Journal best practices (visual distortion)",
    "dual axis": "CPA Journal best practices (misleading correlation)",
    "inconsistent tick intervals": "CPA Journal best practices (distorted perception)",
    "inappropriate axis range": "C&DI 102.10 (misleading presentation)",
    # Completeness checks
    "missing_chart_title": "SEC Staff guidance (clear labeling)",
    "missing_y_axis_label": "SEC Staff guidance (clear labeling)",
    "missing_x_axis_label": "Reg S-K Item 402(v) (Pay-vs-Performance: fiscal year labels required)",
    "missing_y_axis_values": "SEC Staff guidance (readable presentation)",
    "missing_x_axis_values": "Reg S-K Item 402(v) (Pay-vs-Performance: fiscal year labels required)",
    "missing_legend": "SEC Staff guidance (clear labeling)",
    "missing_data_units": "SEC Staff guidance (clear labeling)",
    "missing_data_source": "SEC Staff guidance (data attribution)",
    "missing_nongaap_label": "Reg S-K Item 10(e)(1)(i) (Non-GAAP identification required)",
    "missing_reconciliation_ref": "Reg S-K Item 10(e)(1)(i)(B) (reconciliation to GAAP required)",
    "missing_base_period": "Reg S-K Item 402(v) (indexed comparison base disclosure)",
}

T2_SYSTEM_PROMPT = """You are a financial chart auditor. You detect TWO categories of issues:

PART A — Misleading visual encoding (12 Misviz types): Does the chart visually distort the data?
PART B — Completeness issues (11 types): Is required information missing from the chart?

You have access to tools:

- traditional_ocr: Extract text with bounding boxes from chart regions.
  Use image_id="current". Regions: "full", "y_axis", "x_axis", "title", "bottom", "legend".

- rule_check: Deterministic validation. Use EXACT formats:
  * truncated_axis: {"check_type": "truncated_axis", "data": {"axis_values": [numbers], "chart_type": "bar"}}
  * broken_scale: {"check_type": "broken_scale", "data": {"axis_values": [numbers]}}

- query_memory: Check if data already extracted.

WORKFLOW (max 3 tool calls total):
1. Call traditional_ocr(image_id="current", region="full", mode="bbox") ONCE to get all text.
2. From OCR results, check: Are axis labels present? Are there numeric values? Units? Title? Legend? Source?
3. If numeric Y-axis values found, call rule_check for truncated_axis/broken_scale.
4. Output BOTH Part A (misleaders) and Part B (completeness) in your JSON response.
5. Be precise — only flag issues you have evidence for."""

T2_DETECTION_PROMPT = """Analyze this chart image (chart_id: {chart_id}, page: {page}).

=== PART A: Misleading Visual Encoding ===
Check each of these 12 misleader types:

{misleader_list}

=== PART B: Completeness Issues ===
Check each of these 11 completeness items:

{completeness_list}

PROCEDURE (max 3 tool calls):
1. Call traditional_ocr(image_id="current", region="full", mode="bbox") to get ALL text
2. Call rule_check if numeric axis values found
3. Output your final JSON immediately

Respond with this JSON structure:
{{
  "chart_type": "...",
  "metric_name": "...",
  "is_gaap": true/false,
  "time_window_start": "...",
  "time_window_end": "...",
  "misleaders": {{
    "truncated axis": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    ...
  }},
  "completeness": {{
    "missing_chart_title": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_y_axis_label": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_x_axis_label": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_y_axis_values": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_x_axis_values": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_legend": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_data_units": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_data_source": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_nongaap_label": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_reconciliation_ref": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}},
    "missing_base_period": {{"present": true/false, "confidence": 0.0-1.0, "evidence": "..."}}
  }}
}}"""


def build_detection_prompt(chart_id: str, page: int) -> str:
    """Build the full T2 detection prompt with misleader definitions and completeness checks."""
    misleader_list = "\n".join(
        f"- {name}: {defn}" for name, defn in MISLEADER_DEFINITIONS.items()
    )
    completeness_list = "\n".join(
        f"- {name}: {defn}" for name, defn in COMPLETENESS_CHECKS.items()
    )
    return T2_DETECTION_PROMPT.format(
        chart_id=chart_id, page=page,
        misleader_list=misleader_list,
        completeness_list=completeness_list,
    )
