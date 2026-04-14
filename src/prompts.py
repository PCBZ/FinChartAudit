# src/prompts.py

MISLEADER_TYPES = [
    "misrepresentation",
    "3d",
    "truncated axis",
    "inappropriate use of pie chart",
    "inconsistent tick intervals",
    "dual axis",
    "inconsistent binning size",
    "discretized continuous variable",
    "inappropriate use of line chart",
    "inappropriate item order",
    "inverted axis",
    "inappropriate axis range",
]

TAXONOMY_BLOCK = """
- misrepresentation: bar/area sizes do not match labeled values
- 3d: 3D effects distort visual comparison
- truncated axis: y-axis doesn't start at zero, exaggerating differences
- inappropriate use of pie chart: used for data unsuitable for part-to-whole comparison
- inconsistent tick intervals: axis ticks are unevenly spaced
- dual axis: two y-axes with different scales mislead comparisons
- inconsistent binning size: histogram bins have unequal widths without normalization
- discretized continuous variable: continuous data binned to hide distribution
- inappropriate use of line chart: used for non-sequential or categorical data
- inappropriate item order: ordering creates false impressions
- inverted axis: axis direction reversed, flipping perceived trend
- inappropriate axis range: range set to exaggerate or minimize differences
""".strip()

FEW_SHOT_EXAMPLES = """
EXAMPLE 1
Chart: A bar chart comparing quarterly revenue. The y-axis starts at $800M instead of $0.
Output:
{"misleading": true, "misleader_types": ["truncated axis"], "explanation": "The y-axis begins at $800M rather than zero, visually exaggerating differences between bars."}

EXAMPLE 2
Chart: A pie chart showing year-over-year revenue growth rates ranging from -2% to +15%.
Output:
{"misleading": true, "misleader_types": ["inappropriate use of pie chart"], "explanation": "Growth rates are not parts of a whole and should not be shown as a pie chart."}

EXAMPLE 3
Chart: A line chart showing monthly visits over 12 months, y-axis starts at 0, evenly spaced ticks.
Output:
{"misleading": false, "misleader_types": [], "explanation": "Appropriate axis scaling, consistent ticks, and suitable chart type. No misleading elements detected."}
""".strip()

OUTPUT_FORMAT = """{
  "misleading": <true|false>,
  "misleader_types": [<zero or more types from the taxonomy>],
  "explanation": "<one to three sentences>"
}"""


def build_bbox_text(bboxes: list[dict]) -> str:
    """Convert bbox list from Misviz into a human-readable region description."""
    if not bboxes:
        return "No specific regions flagged."
    lines = ["The following regions have been flagged as potentially problematic:"]
    for i, b in enumerate(bboxes, 1):
        lines.append(
            f"  Region {i}: x={b['x']}, y={b['y']}, width={b['width']}, height={b['height']}"
        )
    lines.append("Pay special attention to these areas when analyzing the chart.")
    return "\n".join(lines)


def build_vision_only_prompt() -> str:
    return f"""You are an expert in data visualization. Detect misleading elements in the chart image.
    
    ## Misleader Taxonomy
    {TAXONOMY_BLOCK}

    ## Examples
    {FEW_SHOT_EXAMPLES}

    ## Output
    Respond with valid JSON only:
    {OUTPUT_FORMAT}"""


def build_vision_text_prompt(ground_truth_text: str) -> str:
    return f"""You are an expert in data visualization. Detect misleading elements by comparing the chart against the ground-truth data.

    ## Misleader Taxonomy
    {TAXONOMY_BLOCK}

    ## Ground-Truth Data
    {ground_truth_text}

    ## Examples
    {FEW_SHOT_EXAMPLES}

    ## Output
    Respond with valid JSON only:
    {OUTPUT_FORMAT}"""


def build_rq3_prompt(sec_context: str) -> str:
    return f"""You are a financial compliance expert. Analyze this SEC filing visual — which may be a chart OR a financial table — for misleading presentation and Non-GAAP prominence violations.

    ## Scope
    Both charts and tables can contain violations:
    - Charts: truncated axes, 3D distortion, inappropriate chart type, etc.
    - Tables: Non-GAAP measures presented more prominently than GAAP measures,
      missing GAAP reconciliation, or non-GAAP figures shown without labeling.

    ## Misleader Taxonomy (for charts)
    {TAXONOMY_BLOCK}

    ## Non-GAAP Prominence Rules (for tables and charts)
    - Non-GAAP measures must NOT appear more prominently than the most directly
      comparable GAAP measure (SEC Regulation G, Item 10(e) of Regulation S-K).
    - Non-GAAP measures must be clearly labeled as such.
    - A reconciliation to the comparable GAAP measure must be provided.
    - Presenting Non-GAAP metrics first, in larger font, or without GAAP context
      constitutes a prominence violation.

    ## SEC Comment Letter Context
    {sec_context}

    ## Output
    Respond with valid JSON only:
    {{
    "misleading": <true|false>,
    "misleader_types": [<zero or more types from the taxonomy, empty list for tables>],
    "sec_violation": "<specific Non-GAAP or chart violation if any, else null>",
    "explanation": "<two to four sentences referencing both the visual and SEC context>"
    }}"""


# ── SEC-specific constants and prompt builders ─────────────────────────────────

SEC_CHART_TAXONOMY = "\n".join(
    f"- {t}: "
    + {
        "truncated axis": "y-axis doesn't start at zero, exaggerating differences",
        "misrepresentation": "bar/area sizes do not match labeled values",
        "3d": "3D effects distort visual comparison",
        "inappropriate use of pie chart": "used for data unsuitable for part-to-whole comparison",
        "dual axis": "two y-axes with different scales mislead comparisons",
        "inconsistent tick intervals": "axis ticks are unevenly spaced",
    }[t]
    for t in [
        "truncated axis",
        "misrepresentation",
        "3d",
        "inappropriate use of pie chart",
        "dual axis",
        "inconsistent tick intervals",
    ]
)

MISREP_VERIFY_PROMPT = """You said this chart has "misrepresentation" — meaning bar/area sizes do NOT match their labeled values.

Please verify: which SPECIFIC bar or element has a visual size that does NOT match its labeled value?
- State the labeled value on that element.
- Describe how its visual size is wrong relative to other elements.

If you cannot identify a specific mismatch, say "no specific mismatch found".

Respond with valid JSON only. Use a boolean for "verified" (true if mismatch confirmed, false otherwise).
Example: {"verified": true, "detail": "Bar labeled $12M is twice as tall as $10M bar"}"""


def build_chart_prompt() -> str:
    """Prompt for SEC chart analysis — 6 relevant types only."""
    return f"""You are a financial compliance expert analyzing a chart from an SEC filing.
Detect misleading visual elements. Only flag clear, unambiguous issues.

## Misleader Taxonomy (check these 6 types only)
{SEC_CHART_TAXONOMY}

## Examples
{FEW_SHOT_EXAMPLES}

## Output
Respond with valid JSON only:
{{
  "misleading": <true|false>,
  "misleader_types": [<zero or more types from the taxonomy above>],
  "explanation": "<one to three sentences>"
}}"""


def build_table_prompt(sec_context: str = "") -> str:
    """Prompt for SEC financial table analysis — Non-GAAP focus."""
    ctx = (
        sec_context
        or "No specific SEC context provided. Analyze the table on its own merits."
    )
    return f"""You are a financial compliance expert. Analyze this financial table from an SEC filing for Non-GAAP prominence violations.

## Non-GAAP Prominence Rules (SEC Regulation G, Item 10(e) of Regulation S-K)
- Non-GAAP measures must NOT appear more prominently than the most directly comparable GAAP measure.
- Non-GAAP measures must be clearly labeled as "Non-GAAP" or "Adjusted" at point of presentation.
- A quantitative reconciliation to the comparable GAAP measure must be provided.
- Presenting Non-GAAP metrics first, in larger font, or without GAAP context = prominence violation.

## What to look for
- Is a Non-GAAP measure (Adjusted EPS, Adjusted EBITDA, Free Cash Flow, etc.) shown more prominently than GAAP?
- Are Non-GAAP measures clearly labeled as such?
- Is there a reconciliation table present?
- Are Non-GAAP figures presented first or in larger/bolder text?

## SEC Comment Letter Context
{ctx}

## Output
Respond with valid JSON only:
{{
  "misleading": <true|false>,
  "sec_violation": "<specific Non-GAAP violation description, or null if none>",
  "explanation": "<two to four sentences>"
}}"""


def build_classify_prompt() -> str:
    """Prompt to classify whether an image is a chart, financial table, or other."""
    return """Look at this image from an SEC 10-K filing.

Classify this image into ONE of these categories:
- "chart" — a data visualization (bar chart, line chart, pie chart, scatter plot, etc.)
- "table" — a financial table with rows and columns of numbers
- "other" — anything else (logo, photo, headshot, signature, decorative image, map, diagram, organizational chart, etc.)

Respond with valid JSON only. Allowed type values: "chart", "table", "other".
Example: {"type": "chart"}"""


def build_table_classify_prompt() -> str:
    """Prompt to classify whether extracted text is a financial data table or something else."""
    return """You are given text extracted from an HTML <table> in an SEC filing.

Classify this content into ONE category:
- "financial_table" — contains actual financial data: income statements, balance sheets, cash flow statements, revenue breakdowns, Non-GAAP reconciliations, segment data, compensation tables with dollar amounts, etc.
- "other" — anything else: legal clauses, plan descriptions, corporate governance text, bullet-point lists, narrative highlights, organizational info, table of contents, signature pages, etc.

Key distinction: a financial table has ROWS of NUMERIC DATA (dollar amounts, percentages, share counts). If the text is mostly prose/paragraphs with a few numbers mentioned in sentences, it is "other".

Respond with valid JSON only. Allowed type values: "financial_table", "other".
Example: {"type": "financial_table", "reason": "Contains income statement line items with dollar amounts"}"""
