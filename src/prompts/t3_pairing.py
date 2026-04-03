"""Prompts for T3 GAAP/Non-GAAP Pairing Agent."""

T3_SYSTEM_PROMPT = """You are an SEC compliance auditor specializing in Non-GAAP financial measures.

Your task: analyze a financial filing to check compliance with SEC Regulation S-K Item 10(e)
and C&DI 100.01-102.13 regarding Non-GAAP measures.

KEY RULES:
1. Every Non-GAAP metric must have a corresponding GAAP metric presented with EQUAL or GREATER prominence.
2. Non-GAAP metrics must be clearly labeled as "Non-GAAP" or "Adjusted".
3. A reconciliation to the most directly comparable GAAP measure must be provided.
4. Non-GAAP measures should not be presented more prominently than GAAP measures.

You have pre-extracted filing data below. Analyze it directly — do NOT call html_extract again.
Focus on validating the Non-GAAP metrics listed in the PRE-EXTRACTED DATA section."""


# Filing type-specific instructions
FILING_TYPE_INSTRUCTIONS = {
    "8-K": (
        "This is an 8-K (Current Report), typically a press release. "
        "Non-GAAP metrics usually appear in the press release body and reconciliation tables. "
        "Check that each Non-GAAP metric has a GAAP counterpart with equal prominence."
    ),
    "10-K": (
        "This is a 10-K (Annual Report). Check MD&A (Item 7) and Financial Highlights "
        "for Non-GAAP/GAAP pairing. Also check if reconciliation tables are present."
    ),
    "DEF14A": (
        "This is a proxy statement (DEF 14A). Check executive compensation metrics "
        "for Non-GAAP usage. Also check TSR charts for completeness."
    ),
}


T3_ANALYSIS_PROMPT = """Analyze this SEC filing for Non-GAAP compliance.

FILING: {file_path}
FILING TYPE: {filing_type}
{filing_type_instruction}

=== PRE-EXTRACTED DATA ===

Source documents analyzed: {source_docs}

Non-GAAP mentions found ({nongaap_count}):
{nongaap_list}

GAAP mentions found ({gaap_count}):
{gaap_list}

Tables found: {table_count}

Filing text ({text_length} chars):
{filing_text}

=== YOUR TASK ===

For EACH Non-GAAP metric listed above:
1. Identify the expected GAAP counterpart:
   - Adjusted EBITDA -> Net Income (absolute amount)
   - Adjusted EBITDA Margin -> Net Income Margin (GAAP ratio, NOT just Net Income amount)
   - Adjusted Operating Income -> Operating Income (GAAP)
   - Adjusted Operating Margin -> Operating Income Margin (GAAP ratio, NOT just Operating Income amount)
   - Non-GAAP EPS / Adjusted EPS -> GAAP EPS (Diluted)
   - Free Cash Flow -> Net Cash from Operations
   - Organic Revenue / Sales -> Total Revenue (GAAP)
2. Check if that EXACT GAAP counterpart appears in the filing with equal prominence.
   CRITICAL: A "margin" (ratio) metric requires a GAAP "margin" counterpart.
   Showing "Net Income" (absolute) does NOT satisfy the pairing for "Adjusted EBITDA Margin" (ratio).
3. Flag any violations.

IMPORTANT: You MUST analyze every Non-GAAP metric listed above. Do not skip any.
IMPORTANT: Only flag violations for metrics presented as STANDALONE KPIs (in headlines,
bullet points, charts, summary tables, or press release highlights).
Do NOT flag descriptive mentions in running text of MD&A (e.g., "organic revenue grew 8%"
as part of a narrative paragraph is normal business discussion, not a standalone Non-GAAP KPI).
When in doubt, check if the metric has its own line/row/heading — if yes, it's a KPI. If it's
embedded in a sentence describing business results, it's a narrative mention.

Respond with ONLY this JSON (no other text):
{{{{
  "metrics": [
    {{{{
      "name": "Adjusted EBITDA",
      "type": "non_gaap",
      "page_or_section": "...",
      "expected_gaap": "Net Income",
      "gaap_found": true,
      "prominence_issue": false,
      "evidence": "..."
    }}}}
  ],
  "pairing_matrix": {{{{
    "total_nongaap": 3,
    "paired": 1,
    "missing": 2,
    "violations": ["Adjusted EBITDA shown without Net Income"]
  }}}}
}}}}"""


def build_t3_prompt(
    file_path: str,
    filing_type: str = "unknown",
    pre_extracted: dict | None = None,
) -> str:
    """Build T3 analysis prompt with pre-extracted filing data.

    Args:
        file_path: Path to the filing (for reference)
        filing_type: Detected filing type ('8-K', '10-K', etc.)
        pre_extracted: Result from HtmlFilingExtractor.extract_filing_complete()
    """
    if pre_extracted is None:
        pre_extracted = {}

    filing_type_instruction = FILING_TYPE_INSTRUCTIONS.get(
        filing_type.upper().replace(" ", ""),
        "Analyze this filing for Non-GAAP compliance.",
    )

    # Format Non-GAAP mentions as a concise list
    nongaap_mentions = pre_extracted.get("nongaap_mentions", [])
    if nongaap_mentions:
        # Deduplicate by term (case-insensitive)
        seen = set()
        unique = []
        for m in nongaap_mentions:
            t = m["term"].lower()
            if t not in seen:
                seen.add(t)
                ctx = m.get("context", "")[:80]
                src = m.get("source", "")
                unique.append(f"  - {m['term']}" + (f" [{src}]" if src else "") + f": ...{ctx}...")
        nongaap_list = "\n".join(unique[:50])  # Cap at 50 unique mentions
    else:
        nongaap_list = "  (none found)"

    gaap_mentions = pre_extracted.get("gaap_mentions", [])
    if gaap_mentions:
        seen = set()
        unique = []
        for m in gaap_mentions:
            t = m["term"].lower()
            if t not in seen:
                seen.add(t)
                unique.append(f"  - {m['term']}")
        gaap_list = "\n".join(unique[:50])
    else:
        gaap_list = "  (none found)"

    # Source documents summary
    source_docs = pre_extracted.get("source_documents", [])
    if source_docs:
        source_docs_str = ", ".join(
            f"{d['type']} ({d['chars']:,} chars, {d.get('nongaap_count', '?')} Non-GAAP)"
            for d in source_docs
        )
    else:
        source_docs_str = "single document"

    # Filing text (already truncated by extractor if needed)
    filing_text = pre_extracted.get("text", "(not available)")

    return T3_ANALYSIS_PROMPT.format(
        file_path=file_path,
        filing_type=filing_type,
        filing_type_instruction=filing_type_instruction,
        source_docs=source_docs_str,
        nongaap_count=len(nongaap_mentions),
        nongaap_list=nongaap_list,
        gaap_count=len(gaap_mentions),
        gaap_list=gaap_list,
        table_count=pre_extracted.get("table_count", 0),
        text_length=pre_extracted.get("text_length", 0),
        filing_text=filing_text,
    )
