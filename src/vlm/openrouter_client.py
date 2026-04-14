"""Shared VLM client for OpenRouter API (OpenAI-compatible).

Extracted from eval_runner.py so both the baseline pipeline and
experiment scripts share a single VLM client implementation.
"""
from __future__ import annotations

import base64
import json
import logging
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import DEFAULT_API_BASE_URL
from src.utils import img_to_data_url  # re-exported for callers' convenience

log = logging.getLogger(__name__)


def make_client(api_key: str, base_url: str = DEFAULT_API_BASE_URL) -> OpenAI:
    """Create an OpenAI-compatible client pointing at OpenRouter."""
    return OpenAI(api_key=api_key, base_url=base_url)


def img_to_b64(img_path: Path, max_bytes: int = 4 * 1024 * 1024) -> tuple[str, str]:
    """Convert image to base64, compressing to JPEG if over max_bytes."""
    img = Image.open(img_path)
    if img.mode in ("CMYK", "RGBA", "P"):
        img = img.convert("RGB")
    quality = 85
    while True:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes or quality <= 30:
            break
        quality -= 10
    buf.seek(0)
    return "image/jpeg", base64.b64encode(buf.read()).decode()


def parse_vlm_response(content: str) -> dict:
    """Parse VLM JSON response, handling markdown code fences."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"misleading": None, "misleader_types": [], "explanation": content, "parse_error": True}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_vlm(client: OpenAI, model: str, prompt: str, image_url: str, max_tokens: int = 512) -> str:
    """Send a vision prompt to the VLM. Retries with exponential backoff."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
        max_tokens=max_tokens,
    )
    if not response.choices or response.choices[0].message.content is None:
        raise ValueError("Empty response from VLM")
    return response.choices[0].message.content


def call_and_parse(client: OpenAI, model: str, prompt: str, image_url: str, max_tokens: int = 512) -> dict:
    """Call VLM and parse JSON response. Returns dict with api_error on failure."""
    try:
        raw = call_vlm(client, model, prompt, image_url, max_tokens)
        return parse_vlm_response(raw)
    except Exception as e:
        log.warning("VLM call failed: %s", e)
        return {"misleading": None, "misleader_types": [], "explanation": str(e), "api_error": True}
