"""HTML filing extractor — parses SEC HTML filings for tables, text, and Non-GAAP mentions.

Supports:
- Single HTML files
- SGML multi-document filings (8-K with embedded Exhibit 99.x)
- Large filings with section-based extraction
- Structure-preserving extraction (headings, bold)
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from pathlib import Path

log = logging.getLogger(__name__)

NONGAAP_PATTERNS = [
    r"\bnon[- ]?gaap\b",
    r"\badjusted\s+(?:ebitda|ebit|operating|net|gross|eps|income|revenue|margin)",
    r"\badj\.\s*(?:ebitda|ebit|operating|net|gross|eps|income|revenue|margin)",
    r"\bcore\s+(?:earnings|income|operating)",
    r"\bfree\s+cash\s+flow\b",
    r"\borganic\s+(?:revenue|growth|sales)",
    r"\bpro\s*forma\b",
    r"\bfunds?\s+from\s+operations\b",
    r"\bconstant\s+currency\b",
    r"\bcomparable\s+(?:sales|store|revenue)",
    r"\bsegment\s+(?:ebitda|operating)",
]
NONGAAP_RE = re.compile("|".join(NONGAAP_PATTERNS), re.IGNORECASE)

GAAP_PATTERNS = [
    r"\bgaap\b",
    r"\bnet\s+(?:income|loss|earnings)\b",
    r"\boperating\s+(?:income|loss)\b",
    r"\bgross\s+profit\b",
    r"\bearnings\s+per\s+share\b",
    r"\beps\b",
    r"\bnet\s+cash\s+(?:provided|used)\b",
    r"\btotal\s+revenue\b",
    r"\bnet\s+(?:revenue|sales)\b",
]
GAAP_RE = re.compile("|".join(GAAP_PATTERNS), re.IGNORECASE)

# Maximum text chars to send to VLM (approx 37K tokens)
MAX_TEXT_CHARS = 150_000

# SGML document boundary pattern (SEC EDGAR multi-document filings)
SGML_DOC_RE = re.compile(
    r"<DOCUMENT>\s*<TYPE>(?P<type>[^\n<]+).*?<TEXT>\s*(?P<body>.*?)</DOCUMENT>",
    re.DOTALL | re.IGNORECASE,
)

# SEC 10-K section heading patterns for sectioning
SECTION_PATTERNS = [
    (r"(?:Item|ITEM)\s+1[.\s]", "item_1_business"),
    (r"(?:Item|ITEM)\s+1A[.\s]", "item_1a_risk"),
    (r"(?:Item|ITEM)\s+7[.\s]", "item_7_mda"),
    (r"(?:Item|ITEM)\s+8[.\s]", "item_8_financials"),
    (r"(?:Item|ITEM)\s+9[.\s]", "item_9"),
    (r"(?i)management.s\s+discussion", "mda"),
    (r"(?i)non[- ]?gaap\s+(?:financial|measures|reconcil)", "nongaap_section"),
    (r"(?i)reconciliation\s+of", "reconciliation"),
    (r"(?i)financial\s+highlights", "financial_highlights"),
]


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: str = ""

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif tag in ("td", "th") and self._in_row:
            self._in_cell = True
            self._current_cell = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            self._current_row.append(self._current_cell.strip())
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if self._current_row:
                self._current_table.append(self._current_row)
        elif tag == "table" and self._in_table:
            self._in_table = False
            if self._current_table:
                self.tables.append(self._current_table)

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell += data


class HtmlFilingExtractor:
    def extract_from_file(self, file_path: str) -> dict:
        html = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return self.extract_from_string(html)

    def extract_from_string(self, html: str, max_text_chars: int = 0) -> dict:
        # Strip XBRL/XML namespace prefixes before tag removal to avoid
        # matching "us-gaap:" fragments as GAAP mentions
        cleaned = re.sub(r"xmlns:\w+=[\"'][^\"']*[\"']", " ", html)
        cleaned = re.sub(r"\bus-gaap:\w+", " ", cleaned)
        cleaned = re.sub(r"\bdei:\w+", " ", cleaned)
        text = re.sub(r"<[^>]+>", " ", cleaned)
        text = re.sub(r"\s+", " ", text).strip()

        parser = _TableParser()
        parser.feed(html)

        # Always scan FULL text for mentions (even if we truncate text for VLM)
        nongaap_mentions = self._find_mentions(text, NONGAAP_RE)
        gaap_mentions = self._find_mentions(text, GAAP_RE)

        # Truncate text if needed (mentions already captured from full text)
        if max_text_chars > 0 and len(text) > max_text_chars:
            text = self._smart_truncate(text, max_text_chars)

        return {
            "text": text,
            "tables": parser.tables,
            "nongaap_mentions": nongaap_mentions,
            "gaap_mentions": gaap_mentions,
            "table_count": len(parser.tables),
            "text_length": len(text),
        }

    # ── SGML multi-document support (8-K with embedded exhibits) ──

    def split_sgml_documents(self, html: str) -> list[dict]:
        """Split an SGML multi-document filing into individual documents.

        SEC EDGAR 8-K filings often embed exhibits using:
            <DOCUMENT><TYPE>EX-99<TEXT>...html...</DOCUMENT>

        Returns list of {"type": "EX-99", "body": "...html..."} dicts.
        Returns empty list if no SGML document boundaries found.
        """
        docs = []
        for m in SGML_DOC_RE.finditer(html):
            docs.append({
                "type": m.group("type").strip(),
                "body": m.group("body").strip(),
            })
        return docs

    def detect_filing_type(self, html: str) -> str:
        """Detect SEC filing type from HTML content.

        Checks SGML header, XBRL metadata, and filename patterns.
        Returns: '8-K', '10-K', '10-Q', 'DEF14A', or 'unknown'.
        """
        head = html[:5000].upper()
        # SGML header: <TYPE>8-K
        type_match = re.search(r"<TYPE>\s*([\w-]+)", head)
        if type_match:
            return type_match.group(1).strip()
        # XBRL: dei:DocumentType
        xbrl_match = re.search(r'dei:documenttype[^>]*>([^<]+)', html[:10000], re.IGNORECASE)
        if xbrl_match:
            return xbrl_match.group(1).strip()
        # Fallback: keywords
        if "FORM 10-K" in head or "ANNUAL REPORT" in head:
            return "10-K"
        if "FORM 8-K" in head or "CURRENT REPORT" in head:
            return "8-K"
        if "DEF 14A" in head or "PROXY STATEMENT" in head:
            return "DEF14A"
        return "unknown"

    def extract_filing_complete(self, file_path: str) -> dict:
        """Extract a complete filing, handling SGML exhibits and large files.

        This is the main entry point for T3 analysis. It:
        1. Detects filing type
        2. Splits SGML documents if present (8-K exhibits)
        3. Extracts and merges text from all relevant documents
        4. Truncates if needed to fit VLM context
        5. Returns unified result with source tracking

        Returns dict with keys:
            text, tables, nongaap_mentions, gaap_mentions,
            filing_type, source_documents, text_length, table_count
        """
        html = Path(file_path).read_text(encoding="utf-8", errors="replace")
        filing_type = self.detect_filing_type(html)
        log.info("Filing %s detected as %s (%d chars)", file_path, filing_type, len(html))

        # Try SGML split first (common for 8-K with embedded exhibits)
        sgml_docs = self.split_sgml_documents(html)

        if sgml_docs:
            return self._extract_from_sgml_docs(sgml_docs, filing_type)

        # Single document — may need sectioning for large files
        result = self.extract_from_string(html, max_text_chars=MAX_TEXT_CHARS)
        result["filing_type"] = filing_type
        result["source_documents"] = [{"type": filing_type, "chars": len(html)}]
        return result

    def _extract_from_sgml_docs(self, docs: list[dict], filing_type: str) -> dict:
        """Extract and merge results from multiple SGML document segments."""
        all_text_parts = []
        all_tables = []
        all_nongaap = []
        all_gaap = []
        source_docs = []

        # Prioritize exhibit documents (EX-99, EX-99.1 etc.) — they have the real content
        # Also include the main 8-K body for context
        for doc in docs:
            doc_type = doc["type"]
            body = doc["body"]

            extracted = self.extract_from_string(body)
            has_financial = len(extracted["nongaap_mentions"]) > 0 or len(extracted["gaap_mentions"]) > 0

            source_docs.append({
                "type": doc_type,
                "chars": len(body),
                "nongaap_count": len(extracted["nongaap_mentions"]),
                "gaap_count": len(extracted["gaap_mentions"]),
                "has_financial": has_financial,
            })

            # Include if it has financial content or is an exhibit
            if has_financial or "EX" in doc_type.upper() or "99" in doc_type:
                offset = sum(len(t) for t in all_text_parts)
                label = f"\n\n=== [{doc_type}] ===\n"
                all_text_parts.append(label)
                all_text_parts.append(extracted["text"])
                all_tables.extend(extracted["tables"])

                # Offset mention positions to account for merged text
                for mention in extracted["nongaap_mentions"]:
                    mention["position"] += offset + len(label)
                    mention["source"] = doc_type
                all_nongaap.extend(extracted["nongaap_mentions"])

                for mention in extracted["gaap_mentions"]:
                    mention["position"] += offset + len(label)
                    mention["source"] = doc_type
                all_gaap.extend(extracted["gaap_mentions"])

        merged_text = "".join(all_text_parts)

        # Truncate if still too large
        if len(merged_text) > MAX_TEXT_CHARS:
            merged_text = self._smart_truncate(merged_text, MAX_TEXT_CHARS)

        # Deduplicate mentions by term (keep first occurrence)
        nongaap_dedup = self._dedup_mentions(all_nongaap)
        gaap_dedup = self._dedup_mentions(all_gaap)

        return {
            "text": merged_text,
            "tables": all_tables,
            "nongaap_mentions": nongaap_dedup,
            "gaap_mentions": gaap_dedup,
            "table_count": len(all_tables),
            "text_length": len(merged_text),
            "filing_type": filing_type,
            "source_documents": source_docs,
        }

    def extract_sections(self, text: str) -> dict[str, str]:
        """Split filing text into SEC sections by heading patterns.

        Returns dict mapping section_id -> section text.
        Falls back to {"full": text} if no sections detected.
        """
        boundaries = []
        for pattern, section_id in SECTION_PATTERNS:
            for m in re.finditer(pattern, text):
                boundaries.append((m.start(), section_id))

        if not boundaries:
            return {"full": text}

        boundaries.sort(key=lambda x: x[0])
        sections = {}
        for i, (start, section_id) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            sections[section_id] = text[start:end]

        return sections

    # ── Helpers ──

    @staticmethod
    def _find_mentions(text: str, regex: re.Pattern) -> list[dict]:
        mentions = []
        for m in regex.finditer(text):
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            mentions.append({
                "term": m.group(),
                "position": m.start(),
                "context": text[start:end].strip(),
            })
        return mentions

    @staticmethod
    def _smart_truncate(text: str, max_chars: int) -> str:
        """Truncate text keeping front and back portions (financial summaries
        are often near the top; reconciliation tables near the end)."""
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return (
            text[:half]
            + f"\n\n[... {len(text) - max_chars:,} chars truncated ...]\n\n"
            + text[-half:]
        )

    @staticmethod
    def _dedup_mentions(mentions: list[dict]) -> list[dict]:
        """Deduplicate mentions, keeping unique (term_lower, source) pairs."""
        seen = set()
        deduped = []
        for m in mentions:
            key = (m["term"].lower(), m.get("source", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(m)
        return deduped

    def run(self, file_path: str) -> dict:
        """Original interface — used by base agent tool executor."""
        return self.extract_from_file(file_path)
