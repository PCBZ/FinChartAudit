# src/config.py
#
# Global configuration for model evaluation, API access, and data downloads

# ── Model & API Configuration ─────────────────────────────────────────────────

DEFAULT_MODELS = {
    "claude": "anthropic/claude-haiku-4.5",
    "qwen": "qwen/qwen3-vl-8b-instruct",
}

DEFAULT_CONDITIONS = ("vision_only", "vision_text")
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"

# ── Download Configuration ────────────────────────────────────────────────────

DEFAULT_USER_AGENT = "FinChartAudit your_email@northeastern.edu"

# SEC EDGAR API endpoints
DEFAULT_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
DEFAULT_SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

# ── ViT veto config (migrated from server.py) ─────────────────────────────────

VIT_VETO_TYPES = {"inconsistent tick intervals"}
VIT_VETO_THRESHOLD = 0.20

# ── SEC pipeline constants (migrated from server.py) ──────────────────────────

SEC_CHART_TYPES = {
    "truncated axis",
    "misrepresentation",
    "3d",
    "inappropriate use of pie chart",
    "dual axis",
    "inconsistent tick intervals",
}
MISREP_DEDUP_TYPES = {"3d", "truncated axis"}
HIGH_SEVERITY_VISUAL = {"truncated axis", "misrepresentation", "3d"}

# ── Resource limits (migrated from server.py) ─────────────────────────────────

MAX_IMAGES = 50
MAX_TABLES = 100
SEC_RATE_LIMIT_SLEEP = 0.15  # SEC EDGAR rate limit compliance

SUPPORTED_FILING_TYPES = ["10-K", "10-Q", "8-K", "DEF 14A", "S-1"]
