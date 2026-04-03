"""DePlot chart-to-table tool — converts chart images to structured data tables.

Uses Google's DePlot (Pix2Struct, 282M params) to extract underlying data from chart images.
Results are cached by image content hash to avoid re-running.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "deplot_cache"


class DePlotTool:
    """Chart-to-table extraction using DePlot."""

    def __init__(self, device: str = "auto", cache_dir: Path | str | None = _DEFAULT_CACHE_DIR):
        self._processor = None
        self._model = None
        self._device = device
        self._available = False
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_hits = 0
        self._cache_misses = 0

    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

            if self._device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._processor = Pix2StructProcessor.from_pretrained("google/deplot")
            self._model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(self._device)
            self._available = True

            # Warm up
            dummy = Image.new("RGB", (100, 100), "white")
            inputs = self._processor(images=dummy, text="Generate underlying data table of the chart below.", return_tensors="pt").to(self._device)
            _ = self._model.generate(**inputs, max_new_tokens=8)

        except Exception as e:
            print(f"DePlot init failed: {e}")
            self._available = False

    def extract_table(self, image_path: str) -> dict:
        """Extract data table from chart image.

        Returns:
            {
                "raw": "raw DePlot output string",
                "title": "chart title if detected",
                "headers": ["col1", "col2", ...],
                "rows": [["val1", "val2", ...], ...],
                "values": [float, ...],  # all numeric values found
            }
        """
        # Check cache
        cached = self._cache_load(image_path)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        self._ensure_loaded()

        if not self._available:
            return {"error": "DePlot not available", "raw": "", "headers": [], "rows": [], "values": []}

        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(
            images=img,
            text="Generate underlying data table of the chart below.",
            return_tensors="pt",
        ).to(self._device)

        predictions = self._model.generate(**inputs, max_new_tokens=512)
        raw = self._processor.decode(predictions[0], skip_special_tokens=True)

        result = self._parse_table(raw)
        self._cache_save(image_path, result)
        return result

    def _parse_table(self, raw: str) -> dict:
        """Parse DePlot's raw output into structured data."""
        lines = [line.strip() for line in raw.split("<0x0A>") if line.strip()]

        title = ""
        headers = []
        rows = []
        all_values = []

        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split("|")]

            if i == 0 and parts[0].upper() == "TITLE":
                title = " | ".join(parts[1:]).strip()
                continue

            if not headers and len(parts) >= 2:
                headers = parts
                continue

            if parts:
                rows.append(parts)

        # Extract all numeric values
        for row in rows:
            for cell in row:
                for num_str in re.findall(r'-?\d+\.?\d*', cell.replace(",", "").replace("%", "")):
                    try:
                        all_values.append(float(num_str))
                    except ValueError:
                        pass

        return {
            "raw": raw,
            "title": title,
            "headers": headers,
            "rows": rows,
            "values": all_values,
        }

    @property
    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._available

    @property
    def cache_stats(self) -> dict:
        return {"hits": self._cache_hits, "misses": self._cache_misses}

    def _cache_key(self, image_path: str) -> str:
        img_bytes = Path(image_path).read_bytes()
        return hashlib.sha256(img_bytes).hexdigest()[:16]

    def _cache_load(self, image_path: str) -> dict | None:
        if not self._cache_dir:
            return None
        key = self._cache_key(image_path)
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _cache_save(self, image_path: str, result: dict) -> None:
        if not self._cache_dir:
            return
        key = self._cache_key(image_path)
        cache_file = self._cache_dir / f"{key}.json"
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
