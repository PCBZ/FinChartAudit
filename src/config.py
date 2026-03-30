# src/config.py
# 
# Global configuration for model evaluation and API access

DEFAULT_MODELS = {
    "claude": "anthropic/claude-haiku-4.5",
    "qwen":   "qwen/qwen3-vl-8b-instruct",
}

DEFAULT_CONDITIONS = ("vision_only", "vision_text")
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
