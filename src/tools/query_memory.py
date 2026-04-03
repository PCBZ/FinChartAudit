"""Memory query tool — agents use this to check cached data before calling OCR."""
from __future__ import annotations

from finchartaudit.memory.filing_memory import FilingMemory


class QueryMemoryTool:
    """Query FilingMemory for previously extracted data."""

    def __init__(self, memory: FilingMemory):
        self.memory = memory

    def run(self, query: str, scope: str = "current_filing") -> dict:
        """Search memory for relevant data.

        Args:
            query: Natural language query or keyword.
            scope: current_filing | historical | all.
        """
        query_lower = query.lower()
        results = {}

        # Search chart registry
        if any(kw in query_lower for kw in ["chart", "registry", "gaap", "non-gaap", "nongaap"]):
            matching_charts = []
            for c in self.memory.chart_registry:
                if (query_lower in c.metric_name.lower()
                        or query_lower in c.chart_type.lower()
                        or ("gaap" in query_lower and not c.is_gaap)
                        or ("nongaap" in query_lower and not c.is_gaap)):
                    matching_charts.append({
                        "chart_id": c.chart_id,
                        "page": c.page,
                        "metric_name": c.metric_name,
                        "is_gaap": c.is_gaap,
                        "chart_type": c.chart_type,
                    })
            if not matching_charts:
                matching_charts = [
                    {"chart_id": c.chart_id, "page": c.page,
                     "metric_name": c.metric_name, "is_gaap": c.is_gaap,
                     "chart_type": c.chart_type}
                    for c in self.memory.chart_registry
                ]
            results["charts"] = matching_charts

        # Search findings
        if any(kw in query_lower for kw in ["finding", "issue", "risk", "problem"]):
            results["findings"] = [
                {"tier": f.tier, "category": f.subcategory,
                 "risk": f.risk_level, "description": f.description[:100]}
                for f in self.memory.findings
            ]

        # Search claims
        if any(kw in query_lower for kw in ["claim", "text", "revenue", "growth", "income"]):
            results["claims"] = [
                {"metric": c.metric, "value": c.value,
                 "text": c.text[:100], "page": c.page}
                for c in self.memory.financial_claims
                if query_lower in c.text.lower() or query_lower in c.metric.lower()
            ]

        # Search OCR cache
        if any(kw in query_lower for kw in ["ocr", "axis", "label", "cached"]):
            cache_keys = [k for k in self.memory.ocr_cache if query_lower in k.lower()]
            results["ocr_cache"] = cache_keys[:10]

        # Search reconciliations
        if any(kw in query_lower for kw in ["reconciliation", "recon"]):
            results["reconciliations"] = self.memory.reconciliations

        if not results:
            results["summary"] = self.memory.get_summary()

        return results
