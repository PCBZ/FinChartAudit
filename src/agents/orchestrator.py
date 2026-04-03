"""Multi-tier orchestrator — dispatches T1, T2, T3 agents and cross-validates."""
from __future__ import annotations

from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.memory.models import AuditFinding, RiskLevel
from finchartaudit.vlm.base import VLMClient


class Orchestrator:
    def __init__(self, vlm: VLMClient, memory: FilingMemory):
        self.vlm = vlm
        self.memory = memory
        self._ocr_tool = None

    def set_ocr_tool(self, ocr_tool):
        self._ocr_tool = ocr_tool

    def audit_chart(self, image_path: str, text_context: str = "",
                    page: int = 0, chart_id: str = "") -> list[AuditFinding]:
        from finchartaudit.agents.t2_visual import T2VisualAgent
        from finchartaudit.agents.t1_numerical import T1NumericalAgent

        all_findings = []
        cid = chart_id or f"p{page}_c1"

        t2 = T2VisualAgent(vlm=self.vlm, memory=self.memory)
        if self._ocr_tool:
            t2.set_ocr_tool(self._ocr_tool)
        all_findings.extend(t2.execute({
            "image_path": image_path, "page": page, "chart_id": cid,
        }))

        if text_context:
            t1 = T1NumericalAgent(vlm=self.vlm, memory=self.memory)
            if self._ocr_tool:
                t1.set_ocr_tool(self._ocr_tool)
            all_findings.extend(t1.execute({
                "image_path": image_path, "text_context": text_context,
                "page": page, "chart_id": cid,
            }))

        return self._cross_validate(all_findings)

    def audit_filing(self, file_path: str, ticker: str = "",
                     filing_type: str = "") -> list[AuditFinding]:
        from finchartaudit.agents.t3_pairing import T3PairingAgent

        t3 = T3PairingAgent(vlm=self.vlm, memory=self.memory)
        t3.execute({
            "file_path": file_path, "ticker": ticker, "filing_type": filing_type,
        })

        all_findings = list(self.memory.findings)
        return self._cross_validate(all_findings)

    def _cross_validate(self, findings: list[AuditFinding]) -> list[AuditFinding]:
        by_chart: dict[str, list[AuditFinding]] = {}
        for f in findings:
            if f.chart_id:
                by_chart.setdefault(f.chart_id, []).append(f)

        for chart_id, chart_findings in by_chart.items():
            tiers = {f.tier for f in chart_findings}
            if len(tiers) >= 2:
                for f in chart_findings:
                    if f.risk_level == RiskLevel.MEDIUM:
                        f.risk_level = RiskLevel.HIGH
                        f.evidence.append(
                            f"Cross-validated: flagged by {len(tiers)} tiers ({', '.join(sorted(tiers))})")

        return findings
