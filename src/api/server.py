"""FinChartAudit API Server — SEC-specific pipeline.

Best pipeline: VLM Call 1 → Rule Dedup → Misrep Re-ask → ViT Classifier Veto.
Separate from experiment scripts (experiments/v*.py) — this is the production API.

Usage:
    pip install -r requirements-api.txt
    python src/api/server.py
"""

from __future__ import annotations

import base64
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from torchvision import transforms

# ── Path setup ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
# Add repo root to sys.path so `from src.data...` imports work when running as script
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("FCA_OPENROUTER_API_KEY") or os.getenv(
    "OPENROUTER_API_KEY", ""
)
VLM_MODEL = os.getenv("FCA_VLM_MODEL", "anthropic/claude-haiku-4.5")
CORS_ORIGINS = os.getenv(
    "FCA_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
VIT_MODEL_PATH = REPO_ROOT / "data" / "models" / "chart_misleader_vit.pt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finchartaudit-api")

# ── Import shared modules ──────────────────────────────────────────────────────

from src.api.sec import fetch_sec_filing, resolve_cik
from src.classifier.model import TYPE_TO_IDX, build_vit_model
from src.config import (
    HIGH_SEVERITY_VISUAL,
    MAX_IMAGES,
    MAX_TABLES,
    MISREP_DEDUP_TYPES,
    SEC_CHART_TYPES,
    SUPPORTED_FILING_TYPES,
    VIT_VETO_THRESHOLD,
    VIT_VETO_TYPES,
)
from src.prompts import (
    MISLEADER_TYPES,
    MISREP_VERIFY_PROMPT,
    SEC_CHART_TAXONOMY,
)
from src.prompts import TAXONOMY_BLOCK as FULL_TAXONOMY_BLOCK
from src.prompts import (
    build_chart_prompt,
    build_classify_prompt,
    build_table_classify_prompt,
    build_table_prompt,
)
from src.utils import img_to_data_url, parse_json

# ── Helpers ───────────────────────────────────────────────────────────────────


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def vlm_call(client, data_url: str, prompt: str, max_tokens: int = 512) -> dict | None:
    """Single VLM call with image. Returns parsed JSON or None. Retries on failure."""
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
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    return parse_json(raw.strip())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def vlm_call_text(client, prompt: str, max_tokens: int = 512) -> dict | None:
    """VLM call with text only (no image). Retries on failure."""
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    return parse_json(raw.strip())


# ── ViT Classifier ────────────────────────────────────────────────────────────


# build_vit_model imported from src.classifier.model

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
                veto_log.append(
                    f"VIT_VETO {t} (prob={prob:.3f} < {VIT_VETO_THRESHOLD})"
                )
                continue
        filtered.append(t)

    return filtered, veto_log


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

    # Initialize shared OpenAI client (reused across requests)
    from openai import OpenAI

    vlm_client = None
    if OPENROUTER_API_KEY:
        vlm_client = OpenAI(
            api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
        )
        log.info(f"OpenAI client initialized (model: {VLM_MODEL})")
    else:
        log.warning("No API key — VLM endpoints will return 500")

    app.state.vit_model = vit_model
    app.state.device = device
    app.state.vlm_client = vlm_client

    yield


app = FastAPI(title="FinChartAudit API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# Global exception handler — ensures CORS headers are always present on error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Request / Response Models ─────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    image_base64: str = Field(
        ..., alias="imageBase64", max_length=20_000_000
    )  # ~15MB image limit
    sec_context: str | None = Field(None, alias="secContext", max_length=20_000)

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
    table_text: str = Field(..., alias="tableText", max_length=100_000)
    sec_context: str | None = Field(None, alias="secContext", max_length=20_000)
    table_name: str | None = Field(None, alias="tableName", max_length=200)

    model_config = {"populate_by_name": True}


class SecFetchRequest(BaseModel):
    ticker: str
    filing_type: str | list[str] = "10-K"
    count: int = Field(
        1, ge=1, le=10
    )  # number of most-recent filings to fetch (pre-filter)
    years: list[int] | None = (
        None  # specific fiscal years to include (e.g. [2024, 2023])
    )


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
def analyze(req: AnalyzeRequest, request: Request):
    client = request.app.state.vlm_client
    if not client:
        raise HTTPException(500, "Server missing FCA_OPENROUTER_API_KEY")

    # Decode image (with decompression bomb guard)
    try:
        image_bytes = base64.b64decode(req.image_base64, validate=True)
        pil_image = Image.open(BytesIO(image_bytes))
        # Guard against decompression bombs
        w, h = pil_image.size
        if w * h > 25_000_000:  # ~25 megapixels
            raise HTTPException(
                400, f"Image too large: {w}x{h} ({w*h/1e6:.0f}MP, max 25MP)"
            )
        pil_image = pil_image.convert("RGB")
        data_url = img_to_data_url(pil_image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    pipeline_log: list[str] = []
    n_calls = 0

    try:
        # ── Step 1: Classify — chart or table? ────────────────────────────
        n_calls += 1
        classify_result = vlm_call(
            client, data_url, build_classify_prompt(), max_tokens=100
        )
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
                raise HTTPException(
                    502, "VLM returned unparseable response for table analysis"
                )

            sec_violation = result.get("sec_violation")
            if isinstance(sec_violation, str) and sec_violation.lower() in (
                "null",
                "none",
                "",
            ):
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
                raise HTTPException(
                    502, "VLM returned unparseable response for chart analysis"
                )

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
                    pipeline_log.append(
                        f"RULE_DEDUP: removed misrep (co-occurs with {overlap})"
                    )

            # ── Step 4: Misrepresentation targeted re-ask ────────────────
            if "misrepresentation" in predicted:
                try:
                    n_calls += 1
                    verify = vlm_call(
                        client, data_url, MISREP_VERIFY_PROMPT, max_tokens=200
                    )
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
            if isinstance(sec_violation, str) and sec_violation.lower() in (
                "null",
                "none",
                "",
            ):
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
    except HTTPException as exc:
        raise
    except Exception as e:
        log.error(f"analyze failed: {e}", exc_info=True)
        raise HTTPException(
            502, f"Analysis pipeline error: {type(e).__name__}: {str(e)[:200]}"
        )


@app.post("/api/analyze-table", response_model=AnalyzeResponse)
def analyze_table(req: AnalyzeTableRequest, request: Request):
    """Analyze a financial table from its text content (no image needed)."""
    client = request.app.state.vlm_client
    if not client:
        raise HTTPException(500, "Server missing FCA_OPENROUTER_API_KEY")

    try:
        pipeline_log = [f"TABLE_TEXT: {(req.table_name or 'unnamed')[:100]}"]
        n_calls = 0

        # ── Step 1: Classify — is this a financial data table or other text?
        n_calls += 1
        classify_prompt = (
            build_table_classify_prompt()
            + f"\n\n## Extracted Text\n{req.table_text[:1500]}"
        )
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
        if isinstance(sec_violation, str) and sec_violation.lower() in (
            "null",
            "none",
            "",
        ):
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
    except HTTPException as exc:
        raise
    except Exception as e:
        log.error(f"analyze-table failed: {e}", exc_info=True)
        raise HTTPException(
            502, f"Table analysis error: {type(e).__name__}: {str(e)[:200]}"
        )


@app.post("/api/sec-fetch")
def sec_fetch(req: SecFetchRequest):
    # Validate ticker format
    if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", req.ticker.upper()):
        raise HTTPException(400, "Invalid ticker format")
    types = req.filing_type if isinstance(req.filing_type, list) else [req.filing_type]
    # Clamp years to prevent excessive downloads (max 10 years)
    years = req.years[:10] if req.years else None
    count = len(years) if years else max(1, min(req.count, 10))
    all_images: list[dict] = []
    all_tables: list[dict] = []
    filing_info: list[dict] = []
    company_name = req.ticker.upper()
    all_doc_context: list[str] = []
    total_comment_letters = 0
    total_skipped = 0

    for ft in types:
        try:
            result = fetch_sec_filing(req.ticker, ft, count=count, years=years)
            all_images.extend(result.get("images", []))
            all_tables.extend(result.get("tables", []))
            filing_info.append(
                {
                    "type": ft,
                    "date": result["filing_date"],
                    "accession": result["accession"],
                }
            )
            if result.get("company_name"):
                company_name = result["company_name"]
            if result.get("doc_context"):
                all_doc_context.append(result["doc_context"])
            total_comment_letters = max(
                total_comment_letters, result.get("comment_letters", 0)
            )
            total_skipped += result.get("skipped_tables", 0)
        except HTTPException as exc:
            if exc.status_code == 404:
                filing_info.append(
                    {"type": ft, "date": None, "error": f"No {ft} found"}
                )
            else:
                raise

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

    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000)
