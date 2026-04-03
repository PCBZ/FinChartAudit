"""Application configuration using pydantic-settings."""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """FinChartAudit configuration. Reads from env vars prefixed with FCA_ or .env file."""

    # OpenRouter API (shared with DSPM project)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Model settings (OpenRouter model IDs)
    vlm_model: str = "anthropic/claude-sonnet-4"   # Primary VLM
    qwen_model: str = "qwen/qwen-2.5-vl-72b-instruct"  # Baseline

    # LLM parameters
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # Detection settings
    confidence_threshold: float = 0.6
    max_reflection_depth: int = 2
    enabled_tiers: list[str] = ["t1", "t2", "t3", "t4"]

    # OCR settings
    ocr_backend: str = "paddleocr"  # "paddleocr" | "rapidocr"

    model_config = {"env_prefix": "FCA_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_config() -> Settings:
    return Settings()
