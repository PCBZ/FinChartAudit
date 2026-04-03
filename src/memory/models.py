"""Core data models used across the system."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Tier(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


# ── Chart & Metric Models ──


@dataclass
class ChartRecord:
    """A chart identified and classified during audit."""
    chart_id: str = ""
    page: int = 0
    metric_name: str = ""
    is_gaap: bool = True
    chart_type: str = ""           # bar, line, pie, 3d_bar, etc.
    axis_origin: float | None = None
    time_window_start: str = ""
    time_window_end: str = ""
    visual_weight: float = 0.0     # 0-1 relative prominence
    font_size_title: float = 0.0
    image_path: str = ""


@dataclass
class PairingEntry:
    """Non-GAAP <-> GAAP pairing status (T3)."""
    nongaap_chart: ChartRecord | None = None
    expected_gaap_metric: str = ""
    gaap_chart: ChartRecord | None = None
    pairing_status: str = ""       # paired | missing | incomplete
    prominence_ratio: float | None = None
    comparability_issues: list[str] = field(default_factory=list)
    reconciliation_page: int | None = None
    risk_level: str = RiskLevel.LOW


@dataclass
class Claim:
    """A numerical claim extracted from narrative text."""
    text: str = ""
    page: int = 0
    metric: str = ""
    value: float | None = None
    context: str = ""


# ── OCR ──


@dataclass
class OCRResult:
    """Cached OCR extraction result."""
    image_id: str = ""
    region: str = "full"
    mode: str = "bbox"
    text_blocks: list[dict] = field(default_factory=list)
    tables: list[dict] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── Audit ──


@dataclass
class TraceEntry:
    """Single step in the audit trace."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent: str = ""
    action: str = ""               # vlm_reasoning | tool_call | tool_result | finding | decision
    tool_name: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    decision: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditFinding:
    """A single audit finding produced by an agent."""
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tier: str = Tier.T2
    category: str = ""             # misleader | text_chart | pairing | cross_section
    subcategory: str = ""          # e.g. truncated_axis, missing_pair
    page: int = 0
    chart_id: str | None = None
    risk_level: str = RiskLevel.LOW
    confidence: float = 0.0
    description: str = ""
    correction: str = ""
    evidence: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    trace: list[TraceEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trace"] = [t.to_dict() if isinstance(t, TraceEntry) else t for t in self.trace]
        return d
