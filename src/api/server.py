"""FinChartAudit API Server — SEC-specific pipeline.

Best pipeline: VLM Call 1 → Rule Dedup → Misrep Re-ask → ViT Classifier Veto.
Separate from experiment scripts (run_pipeline_v*.py) — this is the production API.

Usage:
    pip install -r requirements-api.txt
    python api_server.py
"""

import base64
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import requests as http_requests
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

# ── Path setup ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "finchartaudit"))

load_dotenv(PROJECT_ROOT / ".env")

import os

OPENROUTER_API_KEY = os.getenv("FCA_OPENROUTER_API_KEY", "")
VLM_MODEL = os.getenv("FCA_VLM_MODEL", "anthropic/claude-haiku-4.5")
VIT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "chart_misleader_vit.pt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finchartaudit-api")

# ── Constants ─────────────────────────────────────────────────────────────────

MISLEADER_TYPES = [
    "misrepresentation", "3d", "truncated axis",
    "inappropriate use of pie chart", "inconsistent tick intervals",
    "dual axis", "inconsistent binning size", "discretized continuous variable",
    "inappropriate use of line chart", "inappropriate item order",
    "inverted axis", "inappropriate axis range",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(MISLEADER_TYPES)}

# SEC-relevant types (6 of 12)
SEC_CHART_TYPES = {
    "truncated axis", "misrepresentation", "3d",
    "inappropriate use of pie chart", "dual axis",
    "inconsistent tick intervals",
}

# ViT veto-eligible types and threshold
# NOTE: truncated axis excluded — ViT synth F1=0.14, domain gap causes false vetoes on real SEC charts
VIT_VETO_TYPES = {
    "inconsistent tick intervals",
    "inappropriate axis range", "inappropriate item order",
    "inconsistent binning size",
}
VIT_VETO_THRESHOLD = 0.20

# Types that co-occur with misrepresentation and explain the distortion
MISREP_DEDUP_TYPES = {"3d", "truncated axis"}

# Severity rules
HIGH_SEVERITY_VISUAL = {"truncated axis", "misrepresentation", "3d"}

# ── Prompts ───────────────────────────────────────────────────────────────────

TAXONOMY_BLOCK = """- misrepresentation: bar/area sizes do not match labeled values
- 3d: 3D effects distort visual comparison
- truncated axis: y-axis doesn't start at zero, exaggerating differences
- inappropriate use of pie chart: used for data unsuitable for part-to-whole comparison
- inconsistent tick intervals: axis ticks are unevenly spaced
- dual axis: two y-axes with different scales mislead comparisons"""

SEC_CHART_TAXONOMY = """- truncated axis: y-axis doesn't start at zero, exaggerating differences
- misrepresentation: bar/area sizes do not match labeled values
- 3d: 3D effects distort visual comparison
- inappropriate use of pie chart: used for data unsuitable for part-to-whole comparison
- dual axis: two y-axes with different scales mislead comparisons
- inconsistent tick intervals: axis ticks are unevenly spaced"""

FEW_SHOT_EXAMPLES = """EXAMPLE 1
Chart: A bar chart comparing quarterly revenue. The y-axis starts at $800M instead of $0.
Output:
{"misleading": true, "misleader_types": ["truncated axis"], "explanation": "The y-axis begins at $800M rather than zero, visually exaggerating differences between bars."}

EXAMPLE 2
Chart: A pie chart showing year-over-year revenue growth rates ranging from -2% to +15%.
Output:
{"misleading": true, "misleader_types": ["inappropriate use of pie chart"], "explanation": "Growth rates are not parts of a whole and should not be shown as a pie chart."}

EXAMPLE 3
Chart: A line chart showing monthly visits over 12 months, y-axis starts at 0, evenly spaced ticks.
Output:
{"misleading": false, "misleader_types": [], "explanation": "Appropriate axis scaling, consistent ticks, and suitable chart type. No misleading elements detected."}

EXAMPLE 4
Chart: A 3D bar chart showing quarterly profits. The 3D perspective makes front bars appear larger.
Output:
{"misleading": true, "misleader_types": ["3d"], "explanation": "3D perspective distorts visual comparison of bar sizes."}

EXAMPLE 5
Chart: A line chart showing stock price from $45 to $52 over 6 months. The y-axis starts at $44.
Output:
{"misleading": false, "misleader_types": [], "explanation": "Line charts commonly use non-zero y-axis baselines to show trends clearly."}

EXAMPLE 6
Chart: A bar chart showing satisfaction scores from 4.1 to 4.5 on a scale of 1-5. Y-axis starts at 4.0.
Output:
{"misleading": true, "misleader_types": ["truncated axis"], "explanation": "The bar chart y-axis starts at 4.0 instead of 0, making small differences appear much larger."}"""


def build_chart_prompt() -> str:
    """Prompt for SEC chart analysis — 6 relevant types only."""
    return f"""You are a financial compliance expert analyzing a chart from an SEC filing.
Detect misleading visual elements. Only flag clear, unambiguous issues.

## Misleader Taxonomy (check these 6 types only)
{SEC_CHART_TAXONOMY}

## Examples
{FEW_SHOT_EXAMPLES}

## Output
Respond with valid JSON only:
{{
  "misleading": <true|false>,
  "misleader_types": [<zero or more types from the taxonomy above>],
  "explanation": "<one to three sentences>"
}}"""


def build_table_prompt(sec_context: str = "") -> str:
    """Prompt for SEC financial table analysis — Non-GAAP focus."""
    ctx = sec_context or "No specific SEC context provided. Analyze the table on its own merits."
    return f"""You are a financial compliance expert. Analyze this financial table from an SEC filing for Non-GAAP prominence violations.

## Non-GAAP Prominence Rules (SEC Regulation G, Item 10(e) of Regulation S-K)
- Non-GAAP measures must NOT appear more prominently than the most directly comparable GAAP measure.
- Non-GAAP measures must be clearly labeled as "Non-GAAP" or "Adjusted" at point of presentation.
- A quantitative reconciliation to the comparable GAAP measure must be provided.
- Presenting Non-GAAP metrics first, in larger font, or without GAAP context = prominence violation.

## What to look for
- Is a Non-GAAP measure (Adjusted EPS, Adjusted EBITDA, Free Cash Flow, etc.) shown more prominently than GAAP?
- Are Non-GAAP measures clearly labeled as such?
- Is there a reconciliation table present?
- Are Non-GAAP figures presented first or in larger/bolder text?

## SEC Comment Letter Context
{ctx}

## Output
Respond with valid JSON only:
{{
  "misleading": <true|false>,
  "sec_violation": "<specific Non-GAAP violation description, or null if none>",
  "explanation": "<two to four sentences>"
}}"""


def build_classify_prompt() -> str:
    """Prompt to classify whether an image is a chart, financial table, or other."""
    return """Look at this image from an SEC 10-K filing.

Classify this image into ONE of these categories:
- "chart" — a data visualization (bar chart, line chart, pie chart, scatter plot, etc.)
- "table" — a financial table with rows and columns of numbers
- "other" — anything else (logo, photo, headshot, signature, decorative image, map, diagram, organizational chart, etc.)

Respond with valid JSON only:
{"type": "chart" or "table" or "other"}"""


def build_table_classify_prompt() -> str:
    """Prompt to classify whether extracted text is a financial data table or something else."""
    return """You are given text extracted from an HTML <table> in an SEC filing.

Classify this content into ONE category:
- "financial_table" — contains actual financial data: income statements, balance sheets, cash flow statements, revenue breakdowns, Non-GAAP reconciliations, segment data, compensation tables with dollar amounts, etc.
- "other" — anything else: legal clauses, plan descriptions, corporate governance text, bullet-point lists, narrative highlights, organizational info, table of contents, signature pages, etc.

Key distinction: a financial table has ROWS of NUMERIC DATA (dollar amounts, percentages, share counts). If the text is mostly prose/paragraphs with a few numbers mentioned in sentences, it is "other".

Respond with valid JSON only:
{"type": "financial_table" or "other", "reason": "<one sentence why>"}"""


MISREP_VERIFY_PROMPT = """You said this chart has "misrepresentation" — meaning bar/area sizes do NOT match their labeled values.

Please verify: which SPECIFIC bar or element has a visual size that does NOT match its labeled value?
- State the labeled value on that element.
- Describe how its visual size is wrong relative to other elements.

If you cannot identify a specific mismatch, say "no specific mismatch found".

Respond with valid JSON only:
{"verified": true or false, "detail": "<specific bar and mismatch, or why not verified>"}"""

# ── Helpers ───────────────────────────────────────────────────────────────────


def extract_json(text: str) -> dict | None:
    """Robust JSON extraction: direct parse → fenced block → brace-depth."""
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
            block = text[start:end if end >= 0 else len(text)].strip()
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
                        return json.loads(text[brace : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def img_to_data_url(image_bytes: bytes) -> str:
    """Convert image bytes to JPEG data URL for VLM."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def compute_severity(
    misleading: bool,
    misleader_types: list[str],
    sec_violation: str | None,
) -> str | None:
    if not misleading and not sec_violation:
        return None
    if sec_violation:
        return "HIGH"
    if any(t in HIGH_SEVERITY_VISUAL for t in misleader_types):
        return "HIGH"
    if misleader_types:
        return "MEDIUM"
    return None


# ── VLM Client ────────────────────────────────────────────────────────────────


def vlm_call(client, data_url: str, prompt: str, max_tokens: int = 512) -> dict | None:
    """Single VLM call with image. Returns parsed JSON or None."""
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    return extract_json(raw.strip())


def vlm_call_text(client, prompt: str, max_tokens: int = 512) -> dict | None:
    """VLM call with text only (no image). For HTML table analysis."""
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    return extract_json(raw.strip())


# ── ViT Classifier ────────────────────────────────────────────────────────────


def build_vit_model(num_classes: int = 12) -> nn.Module:
    """Build ViT-B model matching train_classifier.py architecture."""
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    model.head = nn.Sequential(
        nn.Linear(model.num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


VIT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def run_vit_veto(
    pil_image: Image.Image,
    predicted: list[str],
    model: nn.Module,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    """ViT classifier selective veto. Returns (filtered_types, veto_log)."""
    if not model or not predicted:
        return predicted, []

    tensor = VIT_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu()

    veto_log = []
    filtered = []
    for t in predicted:
        if t in VIT_VETO_TYPES and t in TYPE_TO_IDX:
            prob = probs[TYPE_TO_IDX[t]].item()
            if prob < VIT_VETO_THRESHOLD:
                veto_log.append(f"VIT_VETO {t} (prob={prob:.3f} < {VIT_VETO_THRESHOLD})")
                continue
        filtered.append(t)

    return filtered, veto_log


# ── SEC Filing Fetch ──────────────────────────────────────────────────────────

# Import SEC utilities
try:
    from src.data.download_sec_data import SECDownloader
except ImportError:
    SECDownloader = None  # type: ignore

SEC_HEADERS = {
    "User-Agent": "FinChartAudit research@northeastern.edu",
    "Accept-Encoding": "gzip, deflate",
}

# Cache for ticker → (CIK, company_name) lookups
_cik_cache: dict[str, tuple[str, str]] = {}


def resolve_cik(ticker: str) -> tuple[str, str]:
    """Resolve ticker to (CIK, company_name) via SEC EDGAR. Supports any public company."""
    ticker = ticker.upper()

    # Check hardcoded list first (fast path, no name available)
    if SECDownloader and ticker in SECDownloader.COMPANIES:
        return SECDownloader.COMPANIES[ticker], ticker

    # Check cache
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    # Query SEC EDGAR company tickers JSON
    try:
        resp = http_requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS, timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                name = entry.get("title", ticker)
                _cik_cache[ticker] = (cik, name)
                log.info(f"Resolved {ticker} → CIK {cik} ({name})")
                return cik, name
    except Exception as e:
        log.warning(f"SEC ticker lookup failed: {e}")

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


def fetch_sec_filing(ticker: str, filing_type: str = "10-K", count: int = 1,
                     years: list[int] | None = None) -> dict:
    """Download SEC filing(s) and extract chart images + financial table text.

    When years is provided, filters filings to those whose filingDate falls in the given years.
    When count > 1 (and no years filter), downloads the N most recent filings.
    """
    if SECDownloader is None:
        raise HTTPException(503, "SEC downloader module not available — check finchartaudit/src installation")

    filing_type = filing_type.upper()
    if filing_type not in SUPPORTED_FILING_TYPES:
        raise HTTPException(400, f"Unsupported filing type. Valid: {SUPPORTED_FILING_TYPES}")

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
            raise HTTPException(404, f"No {filing_type} filing found for {ticker} in years {sorted(years)}")

    cik_stripped = cik.lstrip("0")
    all_images: list[dict] = []
    all_tables: list[dict] = []
    total_html_tables = 0
    all_context_parts: list[str] = []
    filing_dates: list[str] = []
    filing_accessions: list[str] = []
    seen_filenames: set[str] = set()   # deduplicate images across filings

    for filing in filings:
        acc = filing["accessionNumber"]
        acc_nodash = acc.replace("-", "")
        date = filing["filingDate"]
        doc = filing["primaryDocument"]
        filing_dates.append(date)
        filing_accessions.append(acc)

        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc_nodash}/{doc}"
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
            style = img_tag.get("style", "")
            width = _parse_dim(style, "width") or int(img_tag.get("width", 0) or 0)
            height = _parse_dim(style, "height") or int(img_tag.get("height", 0) or 0)
            # Only skip truly tiny images (icons, spacers)
            if width > 0 and width < 100 and height > 0 and height < 100:
                continue

            src = img_tag.get("src", "")
            filename = Path(src).name
            alt = img_tag.get("alt", "")
            if not filename or filename in seen_filenames:
                continue

            img_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc_nodash}/{filename}"
            try:
                time.sleep(0.15)
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

                display_name = f"{date_tag}_{filename}" if count > 1 else filename
                all_images.append({"name": display_name, "base64": b64, "alt": alt, "type": "chart"})
                seen_filenames.add(filename)
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
            if not _is_candidate_table(table_tag):
                continue
            text = table_tag.get_text(" ", strip=True)[:2000]
            # Clean up the HTML: remove inline styles/classes to keep it small
            for tag in table_tag.find_all(True):
                tag.attrs = {k: v for k, v in tag.attrs.items() if k in ("colspan", "rowspan")}
            table_html = str(table_tag)
            # Cap HTML at 8000 chars to avoid bloating the response
            if len(table_html) > 8000:
                table_html = table_html[:8000] + "<!-- truncated --></table>"
            tbl_name = f"{date_tag}_table_{i:03d}" if count > 1 else f"table_{i:03d}"
            all_tables.append({
                "name": tbl_name, "text": text, "html": table_html,
                "has_nongaap": _has_nongaap_content(text), "type": "table",
            })

        # ── Extract document context text ─────────────────────────────────
        soup_text = BeautifulSoup(html_content, "html.parser")
        for tag in soup_text.find_all(["table", "script", "style"]):
            tag.decompose()
        full_text = soup_text.get_text(" ", strip=True)
        nongaap_kw = ["non-gaap", "adjusted", "reconciliation", "regulation g", "item 10(e)"]
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
    skipped_tables = total_html_tables - len(all_tables)  # non-financial tables filtered by _is_financial_table

    log.info(f"  {len(all_images)} charts, {len(all_tables)} financial tables "
             f"({nongaap_count} with Non-GAAP content, {skipped_tables} non-financial skipped), "
             f"sec context: {len(sec_context)} chars")

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


def _parse_dim(style: str, key: str) -> int:
    m = re.search(rf"{key}\s*:\s*(\d+)\s*(?:px)?", style or "")
    return int(m.group(1)) if m else 0


# ── FastAPI App ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ViT model at startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    vit_model = None
    if VIT_MODEL_PATH.exists():
        log.info(f"Loading ViT model from {VIT_MODEL_PATH}...")
        vit_model = build_vit_model()
        checkpoint = torch.load(VIT_MODEL_PATH, map_location=device, weights_only=True)
        # Handle both checkpoint formats: {model_state_dict: ...} or raw state_dict
        state = checkpoint.get("model_state_dict", checkpoint)
        vit_model.load_state_dict(state)
        vit_model.to(device).eval()
        if "val_f1" in checkpoint:
            log.info(f"ViT model loaded (val_f1={checkpoint['val_f1']:.3f}).")
        else:
            log.info("ViT model loaded.")
    else:
        log.warning(f"ViT model not found at {VIT_MODEL_PATH} — veto disabled.")

    app.state.vit_model = vit_model
    app.state.device = device

    yield


app = FastAPI(title="FinChartAudit API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Global exception handler — ensures CORS headers are always present on error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {str(exc)[:200]}"},
    )


# ── Request / Response Models ─────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    image_base64: str = Field(..., alias="imageBase64", max_length=20_000_000)  # ~15MB image limit
    sec_context: str | None = Field(None, alias="secContext")

    model_config = {"populate_by_name": True}


class AnalyzeResponse(BaseModel):
    misleading: bool
    misleader_types: list[str] = []
    sec_violation: str | None = None
    severity: str | None = None
    rule: str | None = None
    explanation: str = ""
    n_calls: int = 1
    image_type: str = "unknown"
    pipeline_log: list[str] = []


class AnalyzeTableRequest(BaseModel):
    table_text: str = Field(..., alias="tableText")
    sec_context: str | None = Field(None, alias="secContext")
    table_name: str | None = Field(None, alias="tableName")

    model_config = {"populate_by_name": True}


SUPPORTED_FILING_TYPES = ["10-K", "10-Q", "8-K", "DEF 14A", "S-1"]


class SecFetchRequest(BaseModel):
    ticker: str
    filing_type: str | list[str] = "10-K"
    count: int = Field(1, ge=1, le=10)  # number of most-recent filings to fetch (pre-filter)
    years: list[int] | None = None      # specific fiscal years to include (e.g. [2024, 2023])


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "vit_loaded": app.state.vit_model is not None,
        "device": str(app.state.device),
        "model": VLM_MODEL,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "Server missing FCA_OPENROUTER_API_KEY")

    from openai import OpenAI

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    # Decode image
    try:
        image_bytes = base64.b64decode(req.image_base64)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        data_url = img_to_data_url(image_bytes)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    pipeline_log: list[str] = []
    n_calls = 0

    try:
        # ── Step 1: Classify — chart or table? ────────────────────────────
        n_calls += 1
        classify_result = vlm_call(client, data_url, build_classify_prompt(), max_tokens=100)
        image_type = "unknown"
        if classify_result:
            image_type = classify_result.get("type", "unknown")
        pipeline_log.append(f"CLASSIFY: {image_type}")

        # ── Step 2: Route to appropriate pipeline ─────────────────────────

        if image_type not in ("chart", "table"):
            # Not a chart or table — skip analysis (logo, photo, diagram, etc.)
            pipeline_log.append("SKIP: not a chart or table")
            return AnalyzeResponse(
                misleading=False,
                misleader_types=[],
                explanation=f"Image classified as '{image_type}' — not a financial chart or table. No compliance analysis needed.",
                n_calls=n_calls,
                image_type=image_type,
                pipeline_log=pipeline_log,
            )

        if image_type == "table":
            # ── TABLE PIPELINE: Non-GAAP focus ────────────────────────────
            n_calls += 1
            prompt = build_table_prompt(req.sec_context or "")
            result = vlm_call(client, data_url, prompt, max_tokens=600)

            if not result:
                raise HTTPException(502, "VLM returned unparseable response for table analysis")

            sec_violation = result.get("sec_violation")
            if isinstance(sec_violation, str) and sec_violation.lower() in ("null", "none", ""):
                sec_violation = None
            explanation = result.get("explanation", "")
            misleading = bool(result.get("misleading")) or bool(sec_violation)

            return AnalyzeResponse(
                misleading=misleading,
                misleader_types=[],
                sec_violation=sec_violation,
                severity="HIGH" if sec_violation else None,
                rule="Reg G / Item 10(e) S-K" if sec_violation else None,
                explanation=explanation,
                n_calls=n_calls,
                image_type="table",
                pipeline_log=pipeline_log,
            )

        else:
            # ── CHART PIPELINE: 6-type visual detection ──────────────────
            n_calls += 1
            prompt = build_chart_prompt()
            result = vlm_call(client, data_url, prompt, max_tokens=512)

            if not result:
                raise HTTPException(502, "VLM returned unparseable response for chart analysis")

            predicted = result.get("misleader_types", [])
            if isinstance(predicted, str):
                predicted = [predicted]
            predicted = [t for t in predicted if t in SEC_CHART_TYPES]
            explanation = result.get("explanation", "")
            pipeline_log.append(f"VLM_CALL1: {predicted}")

            # ── Step 3: Rule-based dedup (misrep + 3d/truncated) ─────────
            if "misrepresentation" in predicted:
                overlap = set(predicted) & MISREP_DEDUP_TYPES
                if overlap:
                    predicted = [t for t in predicted if t != "misrepresentation"]
                    pipeline_log.append(f"RULE_DEDUP: removed misrep (co-occurs with {overlap})")

            # ── Step 4: Misrepresentation targeted re-ask ────────────────
            if "misrepresentation" in predicted:
                try:
                    n_calls += 1
                    verify = vlm_call(client, data_url, MISREP_VERIFY_PROMPT, max_tokens=200)
                    if verify and not verify.get("verified", True):
                        predicted = [t for t in predicted if t != "misrepresentation"]
                        detail = verify.get("detail", "")[:80]
                        pipeline_log.append(f"MISREP_VETO: unverified ({detail})")
                    else:
                        pipeline_log.append("MISREP_VERIFIED")
                except Exception:
                    pipeline_log.append("MISREP_VERIFY_FAILED (kept)")

            # ── Step 5: ViT Classifier Veto ──────────────────────────────
            predicted, veto_log = run_vit_veto(
                pil_image, predicted, app.state.vit_model, app.state.device
            )
            pipeline_log.extend(veto_log)

            # ── Step 6: Final result ─────────────────────────────────────
            sec_violation = result.get("sec_violation")
            if isinstance(sec_violation, str) and sec_violation.lower() in ("null", "none", ""):
                sec_violation = None

            misleading = len(predicted) > 0 or bool(sec_violation)
            severity = compute_severity(misleading, predicted, sec_violation)

            rule = None
            if sec_violation:
                rule = "Reg G / Item 10(e) S-K"
            elif predicted:
                rule = "Visualization Standards"

            return AnalyzeResponse(
                misleading=misleading,
                misleader_types=predicted,
                sec_violation=sec_violation,
                severity=severity,
                rule=rule,
                explanation=explanation,
                n_calls=n_calls,
                image_type="chart",
                pipeline_log=pipeline_log,
            )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"analyze failed: {e}", exc_info=True)
        raise HTTPException(502, f"Analysis pipeline error: {type(e).__name__}: {str(e)[:200]}")


@app.post("/api/analyze-table", response_model=AnalyzeResponse)
def analyze_table(req: AnalyzeTableRequest):
    """Analyze a financial table from its text content (no image needed)."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "Server missing FCA_OPENROUTER_API_KEY")

    from openai import OpenAI

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    try:
        pipeline_log = [f"TABLE_TEXT: {req.table_name or 'unnamed'}"]
        n_calls = 0

        # ── Step 1: Classify — is this a financial data table or other text?
        n_calls += 1
        classify_prompt = build_table_classify_prompt() + f"\n\n## Extracted Text\n{req.table_text[:1500]}"
        classify_result = vlm_call_text(client, classify_prompt, max_tokens=100)
        table_type = "other"
        if classify_result:
            table_type = classify_result.get("type", "other")
            reason = classify_result.get("reason", "")
            pipeline_log.append(f"CLASSIFY: {table_type} ({reason[:60]})")
        else:
            pipeline_log.append("CLASSIFY: failed, defaulting to other")

        # Skip non-financial content
        if table_type != "financial_table":
            return AnalyzeResponse(
                misleading=False,
                misleader_types=[],
                explanation=f"Content classified as '{table_type}' — not a financial data table. No compliance analysis needed.",
                n_calls=n_calls,
                image_type="other",
                pipeline_log=pipeline_log,
            )

        # ── Step 2: Analyze financial table for Non-GAAP violations
        n_calls += 1
        prompt = build_table_prompt(req.sec_context or "")
        full_prompt = prompt + f"\n\n## Table Content\n{req.table_text[:2000]}"

        result = vlm_call_text(client, full_prompt, max_tokens=500)

        if not result:
            raise HTTPException(502, "VLM returned unparseable response")

        sec_violation = result.get("sec_violation")
        if isinstance(sec_violation, str) and sec_violation.lower() in ("null", "none", ""):
            sec_violation = None
        explanation = result.get("explanation", "")
        misleading = bool(result.get("misleading")) or bool(sec_violation)

        return AnalyzeResponse(
            misleading=misleading,
            misleader_types=[],
            sec_violation=sec_violation,
            severity="HIGH" if sec_violation else None,
            rule="Reg G / Item 10(e) S-K" if sec_violation else None,
            explanation=explanation,
            n_calls=n_calls,
            image_type="table",
            pipeline_log=pipeline_log,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"analyze-table failed: {e}", exc_info=True)
        raise HTTPException(502, f"Table analysis error: {type(e).__name__}: {str(e)[:200]}")


@app.post("/api/sec-fetch")
def sec_fetch(req: SecFetchRequest):
    types = req.filing_type if isinstance(req.filing_type, list) else [req.filing_type]
    # If specific years given, fetch enough filings to cover them; otherwise use count
    count = len(req.years) if req.years else max(1, min(req.count, 10))
    all_images: list[dict] = []
    all_tables: list[dict] = []
    filing_info: list[dict] = []
    company_name = req.ticker.upper()
    all_doc_context: list[str] = []
    total_comment_letters = 0
    total_skipped = 0

    for ft in types:
        try:
            result = fetch_sec_filing(req.ticker, ft, count=count, years=req.years)
            all_images.extend(result.get("images", []))
            all_tables.extend(result.get("tables", []))
            filing_info.append({"type": ft, "date": result["filing_date"], "accession": result["accession"]})
            if result.get("company_name"):
                company_name = result["company_name"]
            if result.get("doc_context"):
                all_doc_context.append(result["doc_context"])
            total_comment_letters = max(total_comment_letters, result.get("comment_letters", 0))
            total_skipped += result.get("skipped_tables", 0)
        except HTTPException:
            filing_info.append({"type": ft, "date": None, "error": f"No {ft} found"})

    return {
        "ticker": req.ticker.upper(),
        "company_name": company_name,
        "filings": filing_info,
        "images": all_images,
        "tables": all_tables,
        "skipped_tables": total_skipped,
        "doc_context": " ".join(all_doc_context)[:10000],
        "comment_letters": total_comment_letters,
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
