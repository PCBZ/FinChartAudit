"""Shared utilities for FinChartAudit."""
from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image


def parse_json(text: str) -> dict | None:
    """Robust JSON extraction: direct parse → fenced block → brace-depth scan.
    Returns None on total failure.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    for marker in ["```json", "```"]:
        idx = text.find(marker)
        if idx >= 0:
            start = idx + len(marker)
            end = text.find("```", start)
            block = text[start: end if end >= 0 else len(text)].strip()
            try:
                return json.loads(block)
            except (json.JSONDecodeError, ValueError):
                pass

    brace = text.find("{")
    if brace >= 0:
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace: i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def img_to_b64(source: Union[str, Path, Image.Image], max_bytes: int = 4 * 1024 * 1024) -> tuple[str, str]:
    """Convert a file path or PIL Image to (mime_type, base64_string) JPEG."""
    if isinstance(source, (str, Path)):
        pil = Image.open(source)
    else:
        pil = source

    if pil.mode in ("CMYK", "RGBA", "P", "LA"):
        pil = pil.convert("RGB")

    quality = 85
    while True:
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes or quality <= 30:
            break
        quality -= 10

    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return "image/jpeg", b64


def img_to_data_url(source: Union[str, Path, Image.Image]) -> str:
    """Return a JPEG data URL suitable for VLM image_url fields."""
    _, b64 = img_to_b64(source)
    return f"data:image/jpeg;base64,{b64}"


def parse_style_dim(style: str, key: str) -> int:
    """Extract px value from inline CSS style string, e.g. 'width:684px' → 684."""
    m = re.search(rf"{key}\s*:\s*(\d+)\s*(?:px)?", style or "")
    return int(m.group(1)) if m else 0
