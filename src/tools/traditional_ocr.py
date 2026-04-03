"""Traditional OCR tool — RapidOCR (default) / PaddleOCR with bounding boxes and caching."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ocr_cache"


class TraditionalOCRTool:
    """OCR with bounding boxes, font size estimation, and disk caching.

    Default backend: RapidOCR (2.5x faster than PaddleOCR on CPU, same quality).
    Cache: results keyed by sha256(image bytes) + region + mode, stored as JSON.
    """

    def __init__(self, backend: str = "rapidocr",
                 cache_dir: Path | str | None = _DEFAULT_CACHE_DIR):
        self._backend = backend
        self._ocr = None
        self._available = False
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_hits = 0
        self._cache_misses = 0
        self._init_ocr()

    def _init_ocr(self):
        if self._backend == "rapidocr":
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr = RapidOCR()
                self._available = True
                return
            except ImportError:
                pass

        # Fallback to PaddleOCR
        try:
            from paddleocr import PaddleOCR
            try:
                self._ocr = PaddleOCR(lang="en", enable_mkldnn=False)
            except (TypeError, ValueError):
                self._ocr = PaddleOCR(lang="en")
            self._backend = "paddleocr"
            self._available = True
            return
        except ImportError:
            pass

        self._available = False

    def run(self, image_path: str, region: str = "full", mode: str = "bbox") -> dict:
        """Run OCR on an image.

        Args:
            image_path: Path to the image file.
            region: Area to OCR — full, y_axis, x_axis, title, bottom, legend.
            mode: Output mode — text, bbox, table, chart_to_table.
        """
        if not self._available:
            return {
                "error": "OCR not installed. pip install rapidocr-onnxruntime or pip install paddleocr",
                "text_blocks": [],
            }

        # Check cache
        cached = self._cache_load(image_path, region, mode)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        img = Image.open(image_path)

        # Crop to region
        if region != "full":
            img = self._crop_region(img, region)

        # Save temp image for OCR
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            temp_path = f.name

        try:
            if self._backend == "paddleocr":
                try:
                    result = self._ocr.ocr(temp_path, cls=True)
                except TypeError:
                    result = self._ocr.ocr(temp_path)
                parsed = self._parse_paddle_result(result, mode)
            else:
                result, _ = self._ocr(temp_path)
                parsed = self._parse_rapid_result(result, mode)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        # Save to cache
        self._cache_save(image_path, region, mode, parsed)
        return parsed

    def _crop_region(self, img: Image.Image, region: str) -> Image.Image:
        w, h = img.size
        crops = {
            "y_axis": (0, 0, int(w * 0.15), h),
            "right_axis": (int(w * 0.85), 0, w, h),
            "x_axis": (0, int(h * 0.85), w, h),
            "title": (0, 0, w, int(h * 0.10)),
            "bottom": (0, int(h * 0.85), w, h),
            "legend": (int(w * 0.80), 0, w, h),
        }
        box = crops.get(region)
        if box:
            return img.crop(box)
        # Try parsing as coordinates: "x1,y1,x2,y2"
        try:
            coords = [int(c) for c in region.split(",")]
            if len(coords) == 4:
                return img.crop(tuple(coords))
        except ValueError:
            pass
        return img

    def _parse_paddle_result(self, result: list, mode: str) -> dict:
        if not result or not result[0]:
            return {"text_blocks": [], "texts": []}

        first = result[0]

        # New PaddleOCR (3.x) returns dict-like OCRResult objects
        if hasattr(first, "keys") and "rec_texts" in first:
            return self._parse_paddle_v3_result(first, mode)

        # Legacy PaddleOCR (2.x) returns list of [box, (text, conf)]
        blocks = []
        texts = []
        for line in first:
            box_points = line[0]
            text = line[1][0]
            confidence = line[1][1]

            x_coords = [p[0] for p in box_points]
            y_coords = [p[1] for p in box_points]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            font_size = bbox[3] - bbox[1]

            texts.append(text)
            blocks.append({
                "text": text,
                "bbox": [round(c, 1) for c in bbox],
                "font_size": round(font_size, 1),
                "confidence": round(confidence, 3),
            })

        if mode == "text":
            return {"texts": texts}
        return {"text_blocks": blocks}

    def _parse_paddle_v3_result(self, ocr_result, mode: str) -> dict:
        """Parse PaddleOCR 3.x dict-like OCRResult object."""
        rec_texts = ocr_result.get("rec_texts", []) or []
        rec_scores = ocr_result.get("rec_scores", []) or []
        rec_polys = ocr_result.get("rec_polys", []) or []
        dt_polys = ocr_result.get("dt_polys", []) or []

        # Use rec_polys if available, otherwise dt_polys
        polys = rec_polys if len(rec_polys) > 0 else dt_polys

        blocks = []
        texts = list(rec_texts)

        for i, text in enumerate(rec_texts):
            confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0

            if i < len(polys):
                poly = polys[i]
                try:
                    # poly can be numpy array or list of points
                    import numpy as np
                    if isinstance(poly, np.ndarray):
                        poly = poly.tolist()
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                except (IndexError, TypeError):
                    bbox = [0, 0, 0, 0]
            else:
                bbox = [0, 0, 0, 0]

            font_size = bbox[3] - bbox[1]

            blocks.append({
                "text": text,
                "bbox": [round(c, 1) for c in bbox],
                "font_size": round(font_size, 1),
                "confidence": round(confidence, 3),
            })

        if mode == "text":
            return {"texts": texts}
        return {"text_blocks": blocks}

    def _parse_rapid_result(self, result: list, mode: str) -> dict:
        if not result:
            return {"text_blocks": [], "texts": []}

        blocks = []
        texts = []
        for item in result:
            bbox = item[0] if len(item) > 0 else []
            text = item[1] if len(item) > 1 else ""
            confidence = item[2] if len(item) > 2 else 0.0

            if isinstance(bbox, list) and len(bbox) >= 4:
                x_coords = [bbox[i][0] for i in range(4)]
                y_coords = [bbox[i][1] for i in range(4)]
                flat_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                font_size = flat_bbox[3] - flat_bbox[1]
            else:
                flat_bbox = [0, 0, 0, 0]
                font_size = 0

            texts.append(text)
            blocks.append({
                "text": text,
                "bbox": flat_bbox,
                "font_size": round(font_size, 1),
                "confidence": round(float(confidence), 3),
            })

        if mode == "text":
            return {"texts": texts}
        return {"text_blocks": blocks}

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def cache_stats(self) -> dict:
        return {"hits": self._cache_hits, "misses": self._cache_misses}

    def _cache_key(self, image_path: str, region: str, mode: str) -> str:
        """Generate cache key from image content hash + region + mode."""
        img_bytes = Path(image_path).read_bytes()
        content_hash = hashlib.sha256(img_bytes).hexdigest()[:16]
        return f"{content_hash}_{region}_{mode}"

    def _cache_load(self, image_path: str, region: str, mode: str) -> dict | None:
        if not self._cache_dir:
            return None
        key = self._cache_key(image_path, region, mode)
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _cache_save(self, image_path: str, region: str, mode: str, result: dict) -> None:
        if not self._cache_dir:
            return
        key = self._cache_key(image_path, region, mode)
        cache_file = self._cache_dir / f"{key}.json"
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
