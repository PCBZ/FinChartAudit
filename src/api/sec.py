"""SEC EDGAR filing fetch utilities — extracted from server.py."""

from __future__ import annotations

import base64
import logging
import threading
import time
from io import BytesIO
from pathlib import Path

import requests as http_requests
from bs4 import BeautifulSoup
from fastapi import HTTPException
from PIL import Image

from src.config import (
    MAX_IMAGES,
    MAX_TABLES,
    SEC_RATE_LIMIT_SLEEP,
    SUPPORTED_FILING_TYPES,
)
from src.utils import parse_style_dim

log = logging.getLogger(__name__)

# Import SEC utilities
try:
    from src.data.download_sec_data import SECDownloader
except ImportError:
    SECDownloader = None  # type: ignore

SEC_HEADERS = {
    "User-Agent": "FinChartAudit research@northeastern.edu",
    "Accept-Encoding": "gzip, deflate",
}

# Cache for ticker → (CIK, company_name) lookups — populated on first miss
_cik_cache: dict[str, tuple[str, str]] = {}
_cik_cache_loaded = False
_cik_cache_lock = threading.Lock()


def _load_cik_cache():
    """Download full SEC ticker→CIK map once and cache in memory."""
    global _cik_cache_loaded
    if _cik_cache_loaded:
        return
    with _cik_cache_lock:
        if _cik_cache_loaded:  # double-check after acquiring lock
            return
        try:
            resp = http_requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=SEC_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            for entry in resp.json().values():
                t = entry.get("ticker", "").upper()
                if t:
                    cik = str(entry["cik_str"]).zfill(10)
                    name = entry.get("title", t)
                    _cik_cache[t] = (cik, name)
            _cik_cache_loaded = True
            log.info(f"Loaded {len(_cik_cache)} tickers from SEC EDGAR")
        except Exception as e:
            log.warning(f"Failed to load SEC ticker map (will retry next request): {e}")


def resolve_cik(ticker: str) -> tuple[str, str]:
    """Resolve ticker to (CIK, company_name) via SEC EDGAR. Supports any public company."""
    ticker = ticker.upper()

    # Check hardcoded list first (fast path, no name available)
    if SECDownloader and ticker in SECDownloader.COMPANIES:
        return SECDownloader.COMPANIES[ticker], ticker

    # Check cache (load full map on first call)
    _load_cik_cache()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    raise HTTPException(404, f"Ticker '{ticker}' not found in SEC EDGAR")


def _is_candidate_table(table_tag) -> bool:
    """Lightweight pre-filter: skip obviously non-data tables. VLM does final classification."""
    text = table_tag.get_text(" ", strip=True)
    # Too short — empty or decorative
    if len(text) < 50:
        return False
    # Must have at least 2 rows
    rows = table_tag.find_all("tr")
    if len(rows) < 2:
        return False
    # Must have at least some cells with content
    cells = table_tag.find_all(["td", "th"])
    if len(cells) < 4:
        return False
    return True


def _has_nongaap_content(text: str) -> bool:
    """Check if table text contains Non-GAAP related content."""
    t = text.lower()
    return any(kw in t for kw in ["non-gaap", "adjusted", "reconciliation"])


def fetch_sec_filing(
    ticker: str,
    filing_type: str = "10-K",
    count: int = 1,
    years: list[int] | None = None,
) -> dict:
    """Download SEC filing(s) and extract chart images + financial table text.

    When years is provided, filters filings to those whose filingDate falls in the given years.
    When count > 1 (and no years filter), downloads the N most recent filings.
    """
    if SECDownloader is None:
        raise HTTPException(
            503,
            "SEC downloader module not available — ensure src.data.download_sec_data is importable (pip install -e '.[api]')",
        )

    filing_type = filing_type.upper()
    if filing_type not in SUPPORTED_FILING_TYPES:
        raise HTTPException(
            400, f"Unsupported filing type. Valid: {SUPPORTED_FILING_TYPES}"
        )

    ticker = ticker.upper()
    cik, company_name = resolve_cik(ticker)
    downloader = SECDownloader()

    # Fetch more than needed so year filter has candidates
    fetch_count = max(count, 10) if years else count
    filings = downloader.get_filings(cik, filing_type, count=fetch_count)
    if not filings:
        raise HTTPException(404, f"No {filing_type} filing found for {ticker}")

    # Filter by specific fiscal years if requested
    if years:
        year_set = set(years)
        filings = [f for f in filings if int(f["filingDate"][:4]) in year_set]
        if not filings:
            raise HTTPException(
                404,
                f"No {filing_type} filing found for {ticker} in years {sorted(years)}",
            )

    cik_stripped = cik.lstrip("0")
    all_images: list[dict] = []
    all_tables: list[dict] = []
    total_html_tables = 0
    all_context_parts: list[str] = []
    filing_dates: list[str] = []
    filing_accessions: list[str] = []
    seen_filenames: set[str] = set()  # deduplicate images across filings

    for filing in filings:
        acc = filing["accessionNumber"]
        acc_nodash = acc.replace("-", "")
        date = filing["filingDate"]
        doc = filing["primaryDocument"]
        filing_dates.append(date)
        filing_accessions.append(acc)

        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc_nodash}/{doc}"
        )
        log.info(f"Downloading {ticker} {filing_type} ({date}) from {filing_url}")

        try:
            resp = http_requests.get(filing_url, headers=SEC_HEADERS, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            log.warning(f"  Failed to download filing {date}: {e}")
            continue
        html_content = resp.text
        soup = BeautifulSoup(html_content, "html.parser")
        date_tag = date[:4]  # year prefix for naming

        # ── Extract images (basic size filter only — VLM classify does final filtering) ──
        imgs = soup.find_all("img")
        for img_tag in imgs:
            if len(all_images) >= MAX_IMAGES:
                break
            style = img_tag.get("style", "")
            width = parse_style_dim(style, "width") or int(img_tag.get("width", 0) or 0)
            height = parse_style_dim(style, "height") or int(
                img_tag.get("height", 0) or 0
            )
            # Only skip truly tiny images (icons, spacers)
            if width > 0 and width < 100 and height > 0 and height < 100:
                continue

            src = img_tag.get("src", "")
            alt = img_tag.get("alt", "")
            if not src or src.startswith("data:"):
                continue
            # Normalize URL: handle absolute, root-relative, and relative paths
            src_clean = src.split("?")[0]
            if src_clean.startswith("http://") or src_clean.startswith("https://"):
                # SSRF guard: only allow sec.gov and its subdomains
                from urllib.parse import urlparse

                parsed_host = (urlparse(src_clean).hostname or "").lower()
                if parsed_host != "sec.gov" and not parsed_host.endswith(".sec.gov"):
                    continue
                img_url = src_clean
            elif src_clean.startswith("/"):
                img_url = f"https://www.sec.gov{src_clean}"
            else:
                img_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc_nodash}/{src_clean}"
            filename = Path(src_clean).name
            dedup_key = f"{acc_nodash}/{src_clean}"
            if not filename or dedup_key in seen_filenames:
                continue
            try:
                time.sleep(SEC_RATE_LIMIT_SLEEP)
                img_resp = http_requests.get(img_url, headers=SEC_HEADERS, timeout=15)
                img_resp.raise_for_status()
                img_bytes = img_resp.content

                pil = Image.open(BytesIO(img_bytes))
                pw, ph = pil.size
                # Skip truly tiny images after download
                if pw < 100 and ph < 100:
                    continue
                if pil.format == "GIF" or pil.mode in ("P", "RGBA", "LA"):
                    pil = pil.convert("RGB")
                buf = BytesIO()
                pil.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()

                display_name = f"{date}_{filename}" if count > 1 else filename
                all_images.append(
                    {"name": display_name, "base64": b64, "alt": alt, "type": "chart"}
                )
                seen_filenames.add(dedup_key)
                log.info(f"  Chart: {display_name} ({len(img_bytes) // 1024}KB)")
            except Exception as e:
                log.warning(f"  Failed to download {filename}: {e}")
                continue

        # ── Extract financial table text (top-level only, skip nested) ────
        html_tables = soup.find_all("table")
        total_html_tables += len(html_tables)
        # Only keep tables that are NOT nested inside another table
        top_tables = [t for t in html_tables if not t.find_parent("table")]
        for i, table_tag in enumerate(top_tables):
            if len(all_tables) >= MAX_TABLES:
                break
            if not _is_candidate_table(table_tag):
                continue
            text = table_tag.get_text(" ", strip=True)[:2000]
            # Clean up the HTML: remove inline styles/classes to keep it small
            for tag in table_tag.find_all(True):
                tag.attrs = {
                    k: v for k, v in tag.attrs.items() if k in ("colspan", "rowspan")
                }
            table_html = str(table_tag)
            # Omit HTML preview for very large tables (avoid broken markup from mid-tag slicing)
            if len(table_html) > 8000:
                table_html = None
            tbl_name = f"{date_tag}_table_{i:03d}" if count > 1 else f"table_{i:03d}"
            all_tables.append(
                {
                    "name": tbl_name,
                    "text": text,
                    "html": table_html,
                    "has_nongaap": _has_nongaap_content(text),
                    "type": "table",
                }
            )

        # ── Extract document context text ─────────────────────────────────
        soup_text = BeautifulSoup(html_content, "html.parser")
        for tag in soup_text.find_all(["table", "script", "style"]):
            tag.decompose()
        full_text = soup_text.get_text(" ", strip=True)
        nongaap_kw = [
            "non-gaap",
            "adjusted",
            "reconciliation",
            "regulation g",
            "item 10(e)",
        ]
        for para in full_text.split("."):
            if any(kw in para.lower() for kw in nongaap_kw):
                all_context_parts.append(para.strip() + ".")

    doc_context = " ".join(all_context_parts)[:8000] if all_context_parts else ""

    # ── Fetch SEC Comment Letters (once per company) ──────────────────────
    comment_context = ""
    comments: list[dict] = []
    try:
        comments = downloader.get_comments(cik, count=5)
        if comments:
            comment_dates = [c["filingDate"] for c in comments]
            comment_context = (
                f"SEC has issued {len(comments)} comment letter(s) to {ticker} "
                f"(dates: {', '.join(comment_dates[:3])}). "
                f"This indicates prior regulatory scrutiny on disclosure practices."
            )
            log.info(f"  Comment letters: {len(comments)} found")
    except Exception as e:
        log.warning(f"  Comment letter fetch failed: {e}")

    sec_context = ""
    if comment_context:
        sec_context += comment_context + " "
    if doc_context:
        sec_context += "Filing context: " + doc_context

    nongaap_count = sum(1 for t in all_tables if t["has_nongaap"])
    skipped_tables = total_html_tables - len(
        all_tables
    )  # non-financial tables filtered by _is_candidate_table

    log.info(
        f"  {len(all_images)} charts, {len(all_tables)} financial tables "
        f"({nongaap_count} with Non-GAAP content, {skipped_tables} non-financial skipped), "
        f"sec context: {len(sec_context)} chars"
    )

    return {
        "ticker": ticker,
        "company_name": company_name,
        "filing_date": filing_dates[0] if filing_dates else "",
        "filing_dates": filing_dates,
        "accession": filing_accessions[0] if filing_accessions else "",
        "images": all_images,
        "tables": all_tables,
        "total_tables": len(all_tables),
        "skipped_tables": skipped_tables,
        "doc_context": sec_context,
        "comment_letters": len(comments) if comments else 0,
    }
