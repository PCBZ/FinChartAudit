"""FilingMemory — central in-memory store for a single filing audit session."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import ChartRecord, PairingEntry, Claim, AuditFinding, OCRResult, RiskLevel, Tier
from .trace import AuditTracer


class FilingMemory:
    """Shared memory for all agents during one filing audit."""

    def __init__(self):
        self.document_map: dict[int, str] = {}  # page -> content type
        self.chart_registry: list[ChartRecord] = []
        self.pairing_matrix: list[PairingEntry] = []
        self.financial_claims: list[Claim] = []
        self.gaap_metrics: dict[str, Any] = {}
        self.nongaap_metrics: dict[str, Any] = {}
        self.reconciliations: dict[str, int] = {}  # metric -> page
        self.ocr_cache: dict[str, OCRResult] = {}
        self.findings: list[AuditFinding] = []
        self.audit_trace = AuditTracer()

    # ── Chart Registry ──

    def register_chart(self, chart: ChartRecord) -> None:
        self.chart_registry.append(chart)

    def get_charts_by_type(self, is_gaap: bool) -> list[ChartRecord]:
        return [c for c in self.chart_registry if c.is_gaap == is_gaap]

    def get_nongaap_charts(self) -> list[ChartRecord]:
        return self.get_charts_by_type(is_gaap=False)

    def get_gaap_charts(self) -> list[ChartRecord]:
        return self.get_charts_by_type(is_gaap=True)

    # ── Findings ──

    def add_finding(self, finding: AuditFinding) -> None:
        self.findings.append(finding)

    def get_findings_by_tier(self, tier: str) -> list[AuditFinding]:
        return [f for f in self.findings if f.tier == tier]

    # ── Claims ──

    def add_claim(self, claim: Claim) -> None:
        self.financial_claims.append(claim)

    # ── OCR Cache ──

    def cache_ocr(self, key: str, result: OCRResult) -> None:
        self.ocr_cache[key] = result

    def get_cached_ocr(self, key: str) -> OCRResult | None:
        return self.ocr_cache.get(key)

    def make_ocr_key(self, image_id: str, region: str, mode: str) -> str:
        return f"{image_id}_{region}_{mode}"

    # ── Summary ──

    def get_summary(self) -> dict:
        risk_counts = {}
        for f in self.findings:
            risk_counts[f.risk_level] = risk_counts.get(f.risk_level, 0) + 1

        tier_counts = {}
        for f in self.findings:
            tier_counts[f.tier] = tier_counts.get(f.tier, 0) + 1

        return {
            "total_charts": len(self.chart_registry),
            "gaap_charts": len(self.get_gaap_charts()),
            "nongaap_charts": len(self.get_nongaap_charts()),
            "total_findings": len(self.findings),
            "findings_by_tier": tier_counts,
            "findings_by_risk": risk_counts,
            "total_claims": len(self.financial_claims),
            "ocr_cache_size": len(self.ocr_cache),
        }

    def export_json(self) -> dict:
        return {
            "summary": self.get_summary(),
            "charts": [asdict(c) for c in self.chart_registry],
            "findings": [f.to_dict() for f in self.findings],
            "claims": [asdict(c) for c in self.financial_claims],
            "trace": self.audit_trace.export_json(),
        }
