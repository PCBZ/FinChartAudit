# src/extract_charts.py
# Extract chart images from SEC 10-K HTM files

import re
import requests
import json
from pathlib import Path
from bs4 import BeautifulSoup
import time
from PIL import Image


HEADERS = {'User-Agent': 'FinChartAudit your_email@northeastern.edu'}
BASE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{filename}"

# minimum size to be considered a chart (not a logo/icon)
MIN_WIDTH  = 200
MIN_HEIGHT = 100


def parse_style_dim(style: str, key: str) -> int:
    """Extract px value from inline style string, e.g. 'width:684px' → 684."""
    m = re.search(rf'{key}\s*:\s*(\d+)', style or '')
    return int(m.group(1)) if m else 0


def is_chart(img_tag) -> bool:
    """Filter out logos/icons based on size."""
    style  = img_tag.get('style', '')
    width  = parse_style_dim(style, 'width')
    height = parse_style_dim(style, 'height')
    # also check explicit width/height attributes
    width  = width  or int(img_tag.get('width',  0) or 0)
    height = height or int(img_tag.get('height', 0) or 0)
    return width >= MIN_WIDTH or height >= MIN_HEIGHT


def convert_gif_to_png(file_path: Path) -> Path:
    """Convert GIF image to PNG. Returns path to PNG file."""
    if file_path.suffix.lower() != '.gif':
        return file_path
    
    try:
        img = Image.open(file_path)
        # Convert to RGB if necessary (handles GIF with transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
        
        png_path = file_path.with_suffix('.png')
        img.save(png_path, 'PNG')
        file_path.unlink()  # Remove original GIF
        return png_path
    except Exception as e:
        print(f"  ⚠ Failed to convert GIF {file_path}: {e}")
        return file_path


def extract_charts_from_htm(
    htm_path: Path,
    cik: str,
    acc_nodash: str,
    out_dir: Path,
) -> list[dict]:
    """
    Parse a 10-K HTM file, find chart images, download and save them.

    Returns list of dicts with chart metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    content = htm_path.read_text(errors='ignore')
    soup    = BeautifulSoup(content, 'html.parser')
    imgs    = soup.find_all('img')

    charts = []
    for img in imgs:
        if not is_chart(img):
            continue

        src      = img.get('src', '')
        alt      = img.get('alt', '')
        filename = Path(src).name

        url = BASE_URL.format(cik=cik, acc_nodash=acc_nodash, filename=filename)
        out_path = out_dir / filename

        if out_path.exists():
            print(f"  ✓ Already exists: {filename}")
        else:
            try:
                time.sleep(0.5)
                resp = requests.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                print(f"  ✓ Downloaded: {filename} ({len(resp.content)//1024} KB)")
            except Exception as e:
                print(f"  ✗ Failed {filename}: {e}")
                continue

        # Convert GIF to PNG
        out_path = convert_gif_to_png(out_path)
        filename = out_path.name

        charts.append({
            'filename': filename,
            'alt':      alt,
            'path':     str(out_path),
            'url':      url,
        })

    return charts


def extract_all(
    sec_dir:  str = 'data/sec',
    pdfs_dir: str = 'data/pdfs',
    out_dir:  str = 'data/charts',
):
    """Extract charts from all downloaded 10-K HTM files."""
    all_charts = {}

    for meta_path in sorted(Path(sec_dir).glob('*.json')):
        meta   = json.loads(meta_path.read_text())
        ticker = meta['ticker']
        cik    = meta['cik'].lstrip('0')

        all_charts[ticker] = []

        for filing in meta.get('filings_10k', []):
            acc       = filing['accessionNumber']
            acc_nodash = acc.replace('-', '')
            date      = filing['filingDate']
            doc       = filing['primaryDocument']

            htm_path = Path(pdfs_dir) / ticker / f"{date}_{doc}"
            if not htm_path.exists():
                print(f"  ✗ HTM not found: {htm_path}")
                continue

            chart_out = Path(out_dir) / ticker / acc_nodash
            print(f"\n{ticker} | {date} | {acc}")

            charts = extract_charts_from_htm(
                htm_path=htm_path,
                cik=cik,
                acc_nodash=acc_nodash,
                out_dir=chart_out,
            )
            print(f"  → {len(charts)} chart(s) found")

            for c in charts:
                all_charts[ticker].append({
                    'date':       date,
                    'accession':  acc,
                    **c,
                })

    # save manifest
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    manifest_path.write_text(json.dumps(all_charts, indent=2))
    print(f"\n💾 Manifest saved → {manifest_path}")

    total = sum(len(v) for v in all_charts.values())
    print(f"📊 Total charts extracted: {total}")
    return all_charts


if __name__ == "__main__":
    extract_all()