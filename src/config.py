# src/config.py
# 
# Global configuration for model evaluation, API access, and data downloads

# ── Model & API Configuration ─────────────────────────────────────────────────

DEFAULT_MODELS = {
    "claude": "anthropic/claude-haiku-4.5",
    "qwen":   "qwen/qwen3-vl-8b-instruct",
}

DEFAULT_CONDITIONS = ("vision_only", "vision_text")
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"

# ── Download Configuration ────────────────────────────────────────────────────

DEFAULT_USER_AGENT = "FinChartAudit your_email@northeastern.edu"

# SEC EDGAR API endpoints
DEFAULT_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
DEFAULT_SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
