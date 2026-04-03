"""PDF text extraction tool — wraps PyMuPDF for zero-error embedded text."""
from __future__ import annotations

from pathlib import Path

import fitz


class ExtractPdfTextTool:
    """Extract embedded text and tables from PDF pages."""

    def __init__(self, pdf_path: str | Path):
        self.doc = fitz.open(str(pdf_path))

    def run(self, page: int, extract_tables: bool = True) -> dict:
        """Extract text (and optionally tables) from a page.

        Args:
            page: 1-indexed page number.
            extract_tables: Whether to also extract table structures.

        Returns:
            dict with text, tables, page, has_content.
        """
        idx = page - 1
        if idx < 0 or idx >= len(self.doc):
            return {"text": "", "tables": None, "page": page, "has_content": False,
                    "error": f"Page {page} out of range (1-{len(self.doc)})"}

        pg = self.doc[idx]
        text = pg.get_text("text")

        tables = None
        if extract_tables:
            raw_tables = pg.find_tables()
            if raw_tables:
                tables = []
                for tbl in raw_tables:
                    rows = []
                    for row in tbl.extract():
                        rows.append([cell if cell else "" for cell in row])
                    tables.append(rows)

        return {
            "text": text,
            "tables": tables,
            "page": page,
            "has_content": len(text.strip()) > 50,
        }

    def close(self):
        self.doc.close()
