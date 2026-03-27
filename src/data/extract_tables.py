# src/extract_tables.py
# Extract financial table screenshots from SEC 10-K HTM files using Playwright

import json
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# Minimum table size to be considered a financial table (not nav/header)
MIN_WIDTH  = 300
MIN_HEIGHT = 80
MIN_ROWS   = 2


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

        # find all tables
        table_elements = await page.query_selector_all("table")
        print(f"  Found {len(table_elements)} tables in {htm_path.name}")

        for i, table in enumerate(table_elements):
            box = await table.bounding_box()
            if not box:
                continue

            w, h = box["width"], box["height"]
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                continue

            # check row count
            rows = await table.query_selector_all("tr")
            if len(rows) < MIN_ROWS:
                continue

            # screenshot the table element
            fname = f"table_{i:03d}.png"
            out_path = out_dir / fname
            await table.screenshot(path=str(out_path))

            tables.append({
                "filename": fname,
                "path":     str(out_path),
                "ticker":   ticker,
                "date":     date,
                "table_idx": i,
                "width":    round(w),
                "height":   round(h),
                "n_rows":   len(rows),
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
            date = filing["filingDate"]
            doc  = filing["primaryDocument"]
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

    # save manifest
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