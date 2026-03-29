# src/extract_tables.py
# Extract financial table screenshots from SEC 10-K HTM files using Playwright

import json
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# Minimum table size
MIN_WIDTH  = 400   # wider than before — nav tables are usually narrow
MIN_HEIGHT = 120   # taller than before — real financial tables have more rows
MIN_ROWS   = 4     # at least 4 rows — filters out simple 2-row headers
MIN_COLS   = 2     # at least 2 columns

# Financial keyword signals — table must contain at least one
FINANCIAL_KEYWORDS = [
    # income statement
    "revenue", "net income", "net loss", "earnings", "ebitda", "gross profit",
    "operating income", "operating loss", "cost of revenue", "cost of sales",
    # balance sheet
    "total assets", "total liabilities", "stockholders", "equity", "cash and cash",
    # cash flow
    "cash flow", "capital expenditure", "free cash flow",
    # non-gaap
    "non-gaap", "adjusted", "reconciliation",
    # per share
    "per share", "diluted", "basic",
    # general financial
    "fiscal", "quarter", "annual", "segment", "million", "billion",
    "gaap", "margin", "growth", "variance",
]

# Keywords that indicate NON-financial tables to skip
SKIP_KEYWORDS = [
    "exhibit", "signature", "power of attorney", "index to",
    "table of contents", "part i", "part ii", "part iii", "part iv",
]


async def _is_financial_table(table, page) -> bool:
    """Check if table contains financial keywords in its text content."""
    try:
        text = await table.inner_text()
        text_lower = text.lower()

        # skip if contains nav/structural keywords
        if any(kw in text_lower for kw in SKIP_KEYWORDS):
            return False

        # must contain at least one financial keyword
        return any(kw in text_lower for kw in FINANCIAL_KEYWORDS)
    except Exception:
        return False


async def extract_tables_from_htm(htm_path: Path, out_dir: Path, ticker: str, date: str) -> list[dict]:
    """
    Render a 10-K HTM file and screenshot each financial table.
    Returns list of dicts with table metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page    = await browser.new_page(viewport={"width": 1400, "height": 900})

        await page.goto(f"file://{htm_path.resolve()}", wait_until="domcontentloaded")

        table_elements = await page.query_selector_all("table")
        print(f"  Found {len(table_elements)} raw tables in {htm_path.name}")

        for i, table in enumerate(table_elements):
            box = await table.bounding_box()
            if not box:
                continue

            w, h = box["width"], box["height"]
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                continue

            rows = await table.query_selector_all("tr")
            if len(rows) < MIN_ROWS:
                continue

            # check column count from first row
            first_row = rows[0]
            cols = await first_row.query_selector_all("td, th")
            if len(cols) < MIN_COLS:
                continue

            # financial keyword filter
            if not await _is_financial_table(table, page):
                continue

            fname    = f"table_{i:03d}.png"
            out_path = out_dir / fname
            await table.screenshot(path=str(out_path))

            tables.append({
                "filename":  fname,
                "path":      str(out_path),
                "alt":       fname,
                "ticker":    ticker,
                "date":      date,
                "table_idx": i,
                "width":     round(w),
                "height":    round(h),
                "n_rows":    len(rows),
                "n_cols":    len(cols),
            })

        await browser.close()

    print(f"  → {len(tables)} financial tables saved")
    return tables


def extract_all(
    sec_dir:  str = "data/sec",
    pdfs_dir: str = "data/pdfs",
    out_dir:  str = "data/tables",
):
    """Extract tables from all downloaded 10-K HTM files."""
    all_tables = {}

    for meta_path in sorted(Path(sec_dir).glob("*.json")):
        meta   = json.loads(meta_path.read_text())
        ticker = meta["ticker"]
        all_tables[ticker] = []

        for filing in meta.get("filings_10k", []):
            date     = filing["filingDate"]
            doc      = filing["primaryDocument"]
            htm_path = Path(pdfs_dir) / ticker / f"{date}_{doc}"

            if not htm_path.exists():
                print(f"  ✗ HTM not found: {htm_path}")
                continue

            table_out = Path(out_dir) / ticker / date
            print(f"\n{ticker} | {date}")

            extracted = asyncio.run(extract_tables_from_htm(
                htm_path=htm_path,
                out_dir=table_out,
                ticker=ticker,
                date=date,
            ))

            all_tables[ticker].extend(extracted)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(all_tables, indent=2))

    total = sum(len(v) for v in all_tables.values())
    print(f"\n💾 Manifest saved → {manifest_path}")
    print(f"📊 Total tables extracted: {total}")
    return all_tables


if __name__ == "__main__":
    extract_all()