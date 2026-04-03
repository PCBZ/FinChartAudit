"""PDF parser — extract pages, charts, text, sections from SEC filings."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import fitz  # PyMuPDF


class FilingParser:
    """Parse SEC filing PDFs into structured components."""

    SECTION_PATTERNS = [
        (r"(?i)item\s*1[.\s]+business", "business"),
        (r"(?i)item\s*1a[.\s]+risk\s*factors", "risk_factors"),
        (r"(?i)item\s*7[.\s]+management.s\s*discussion", "mda"),
        (r"(?i)item\s*8[.\s]+financial\s*statements", "financial_statements"),
        (r"(?i)non[- ]?gaap", "nongaap_disclosure"),
    ]

    MIN_CHART_WIDTH = 200
    MIN_CHART_HEIGHT = 150

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(str(self.pdf_path))
        self._temp_dir = Path(tempfile.mkdtemp(prefix="fca_"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.doc.close()

    @property
    def page_count(self) -> int:
        return len(self.doc)

    def extract_text(self, page: int) -> str:
        """Extract text from a 1-indexed page."""
        return self.doc[page - 1].get_text("text")

    def extract_tables(self, page: int) -> list[list[list[str]]]:
        """Extract tables from a 1-indexed page."""
        tables = self.doc[page - 1].find_tables()
        result = []
        for tbl in tables:
            rows = []
            for row in tbl.extract():
                rows.append([cell if cell else "" for cell in row])
            result.append(rows)
        return result

    def render_page(self, page: int, dpi: int = 200) -> bytes:
        """Render a page as PNG bytes."""
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = self.doc[page - 1].get_pixmap(matrix=mat)
        return pix.tobytes("png")

    def save_page_image(self, page: int, dpi: int = 200) -> str:
        """Render page and save to temp file. Returns path."""
        img_bytes = self.render_page(page, dpi)
        path = self._temp_dir / f"page_{page}.png"
        path.write_bytes(img_bytes)
        return str(path)

    def extract_charts(self) -> list[dict]:
        """Extract chart-sized images from all pages."""
        charts = []
        for page_num in range(self.page_count):
            page = self.doc[page_num]
            images = page.get_images(full=True)

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    base_image = self.doc.extract_image(xref)
                except Exception:
                    continue

                w = base_image.get("width", 0)
                h = base_image.get("height", 0)
                if w < self.MIN_CHART_WIDTH or h < self.MIN_CHART_HEIGHT:
                    continue

                chart_id = f"p{page_num + 1}_c{img_idx + 1}"
                ext = base_image.get("ext", "png")
                img_path = self._temp_dir / f"{chart_id}.{ext}"
                img_path.write_bytes(base_image["image"])

                text = page.get_text("text")

                charts.append({
                    "chart_id": chart_id,
                    "page": page_num + 1,
                    "image_path": str(img_path),
                    "width": w,
                    "height": h,
                    "surrounding_text": text[:500],
                })

        return charts

    def detect_sections(self) -> list[dict]:
        """Detect 10-K sections by heading patterns."""
        sections = []
        current = None

        for page_num in range(self.page_count):
            text = self.doc[page_num].get_text("text")
            for pattern, section_name in self.SECTION_PATTERNS:
                if re.search(pattern, text):
                    if current:
                        current["end_page"] = page_num + 1
                        sections.append(current)
                    match = re.search(pattern, text)
                    line_start = text.rfind("\n", 0, match.start()) + 1
                    line_end = text.find("\n", match.end())
                    title = text[line_start:line_end].strip() if line_end > 0 else ""
                    current = {
                        "name": section_name,
                        "title": title[:100],
                        "start_page": page_num + 1,
                        "end_page": self.page_count,
                    }
                    break

        if current:
            sections.append(current)
        return sections

    def get_section_text(self, section_name: str) -> str:
        """Get full text for a named section."""
        for section in self.detect_sections():
            if section["name"] == section_name:
                pages = []
                for p in range(section["start_page"], section["end_page"] + 1):
                    if p <= self.page_count:
                        pages.append(self.extract_text(p))
                return "\n\n".join(pages)
        return ""
