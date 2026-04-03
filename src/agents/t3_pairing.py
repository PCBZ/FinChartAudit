"""T3 GAAP/Non-GAAP Pairing Agent — detects prominence and pairing violations."""
from __future__ import annotations

import json
import logging

from finchartaudit.agents.base import BaseAgent
from finchartaudit.memory.models import AuditFinding, PairingEntry, RiskLevel, Tier
from finchartaudit.prompts.t3_pairing import T3_SYSTEM_PROMPT, build_t3_prompt
from finchartaudit.tools.html_extract import HtmlFilingExtractor

log = logging.getLogger(__name__)


NONGAAP_TO_GAAP = {
    "adjusted ebitda": "net income",
    "adjusted ebitda margin": "net income margin",
    "adjusted operating income": "operating income",
    "adjusted operating margin": "operating income margin",
    "adjusted net income": "net income",
    "adjusted eps": "earnings per share",
    "non-gaap eps": "earnings per share",
    "non-gaap operating income": "operating income",
    "free cash flow": "net cash from operations",
    "organic revenue": "total revenue",
    "organic sales": "total revenue",
    "organic growth": "total revenue",
    "core operating income": "operating income",
    "adjusted gross margin": "gross margin",
    "adjusted ebit": "operating income",
    "adjusted ebit margin": "operating income margin",
    "adj. ebitda": "net income",
    "funds from operations": "net income",
    "comparable store sales": "total revenue",
    "constant currency revenue": "total revenue",
}


class T3PairingAgent(BaseAgent):
    agent_name = "t3_pairing"
    available_tools = ["rule_check", "query_memory"]  # html_extract no longer needed as tool

    def execute(self, task: dict) -> list[AuditFinding]:
        file_path = task["file_path"]
        ticker = task.get("ticker", "")
        page = task.get("page", 0)

        # Phase 1: Pre-extract filing data (handles exhibits, sectioning, truncation)
        extractor = HtmlFilingExtractor()
        pre_extracted = extractor.extract_filing_complete(file_path)
        filing_type = pre_extracted.get("filing_type", "unknown")
        nongaap_count = len(pre_extracted.get("nongaap_mentions", []))

        log.info("[%s] %s: %s, %d Non-GAAP mentions, %d chars",
                 ticker, filing_type,
                 [d["type"] for d in pre_extracted.get("source_documents", [])],
                 nongaap_count, pre_extracted.get("text_length", 0))

        self.memory.audit_trace.log_tool_call(
            self.agent_name, "extract_filing_complete",
            f"{filing_type}, {nongaap_count} nongaap, "
            f"{len(pre_extracted.get('source_documents', []))} docs")

        # Phase 2: Build prompt with pre-extracted data and call VLM
        prompt = build_t3_prompt(
            file_path=file_path,
            filing_type=filing_type,
            pre_extracted=pre_extracted,
        )

        # Single VLM call — no tool-use loop needed since data is pre-extracted
        response = self.vlm.analyze(
            image_path="", prompt=prompt, tools=None, system=T3_SYSTEM_PROMPT)
        final_text = response.text

        if final_text:
            self.memory.audit_trace.log_reasoning(self.agent_name, final_text[:300])

        # Phase 3: Parse response
        json_data = self._extract_json(final_text)
        findings = self._build_findings(json_data, ticker, page) if json_data else []

        # Phase 3.5: Retry ONLY if JSON parsing actually failed (not if VLM found no violations)
        if json_data is None and nongaap_count > 0:
            log.warning("[%s] JSON parse failed with %d Non-GAAP mentions — retrying",
                        ticker, nongaap_count)

            terms = list({m["term"].lower() for m in pre_extracted["nongaap_mentions"]})
            retry_prompt = (
                f"Your previous response could not be parsed as valid JSON. "
                f"The filing contains {nongaap_count} Non-GAAP mentions: {', '.join(terms[:20])}. "
                f"Please respond with ONLY the JSON object in the exact format specified, "
                f"with no additional text before or after the JSON."
            )
            retry_response = self.vlm.analyze(
                image_path="", prompt=retry_prompt, tools=None, system=T3_SYSTEM_PROMPT)
            if retry_response.text:
                self.memory.audit_trace.log_reasoning(
                    self.agent_name, f"RETRY_RESPONSE: {retry_response.text[:300]}")
                json_data = self._extract_json(retry_response.text)
                if json_data:
                    findings = self._build_findings(json_data, ticker, page)

        return findings

    def _build_findings(self, json_data: dict, ticker: str, page: int) -> list[AuditFinding]:
        """Build AuditFinding objects from parsed VLM JSON response."""
        findings = []

        for metric in json_data.get("metrics", []):
            if metric.get("type") != "non_gaap":
                continue

            name = metric.get("name", "")
            gaap_found = metric.get("gaap_found", True)
            prominence_issue = metric.get("prominence_issue", False)

            pairing = PairingEntry(
                expected_gaap_metric=metric.get("expected_gaap", ""),
                pairing_status="paired" if gaap_found else "missing",
            )
            self.memory.pairing_matrix.append(pairing)

            if not gaap_found:
                findings.append(AuditFinding(
                    tier=Tier.T3,
                    category="pairing",
                    subcategory="missing_gaap_counterpart",
                    page=page,
                    risk_level=RiskLevel.HIGH,
                    confidence=0.8,
                    description=(
                        f"{name} presented without corresponding GAAP metric "
                        f"({metric.get('expected_gaap', 'unknown')})"
                    ),
                    correction=(
                        f"Present {metric.get('expected_gaap', 'GAAP counterpart')} "
                        f"with equal or greater prominence alongside {name}."
                    ),
                    evidence=[
                        metric.get("evidence", ""),
                        "SEC basis: Reg S-K Item 10(e)(1)(i)(A) — GAAP comparison required",
                    ],
                ))

            if prominence_issue:
                findings.append(AuditFinding(
                    tier=Tier.T3,
                    category="pairing",
                    subcategory="undue_prominence",
                    page=page,
                    risk_level=RiskLevel.HIGH,
                    confidence=0.7,
                    description=f"{name} has undue prominence over GAAP counterpart",
                    correction="Ensure Non-GAAP measures are not presented more prominently than GAAP.",
                    evidence=[
                        metric.get("evidence", ""),
                        "SEC basis: C&DI 102.10 — Non-GAAP must not have undue prominence",
                    ],
                ))

        pairing_data = json_data.get("pairing_matrix", {})
        for violation in pairing_data.get("violations", []):
            if not any(violation in f.description for f in findings):
                findings.append(AuditFinding(
                    tier=Tier.T3,
                    category="pairing",
                    subcategory="pairing_violation",
                    page=page,
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.6,
                    description=violation,
                    correction="Ensure all Non-GAAP metrics have corresponding GAAP measures.",
                    evidence=["SEC basis: Reg S-K Item 10(e)"],
                ))

        for f in findings:
            self.memory.add_finding(f)
            self.memory.audit_trace.log_finding(
                self.agent_name, f"{f.subcategory}: {f.description[:80]}")

        return findings

    def _extract_json(self, text: str) -> dict | None:
        if not text:
            log.warning("[%s] Empty VLM response", self.agent_name)
            return None

        # Attempt 1: direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Attempt 2: markdown code block
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        # Attempt 3: brace matching
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

        # All attempts failed — log for debugging
        log.warning("[%s] JSON parse failed. Raw text (%d chars): %s",
                    self.agent_name, len(text), text[:500])
        self.memory.audit_trace.log_reasoning(
            self.agent_name, f"JSON_PARSE_FAIL ({len(text)} chars): {text[:300]}")
        return None
