# src/eval_runner.py

import json
import os
import time
import base64
from pathlib import Path
from io import BytesIO

from openai import OpenAI
from huggingface_hub import login
from datasets import load_dataset

from tqdm import tqdm
from prompts import (
    build_vision_only_prompt,
    build_vision_text_prompt,
    build_bbox_text,
    build_rq3_prompt,
    MISLEADER_TYPES,
)

# ── Model configurations ──────────────────────────────────────────────────────

MODELS = {
    "claude": "anthropic/claude-sonnet-4-5",
    "qwen":   "qwen/qwen3-vl-235b-a22b-instruct",
}

CONDITIONS = ("vision_only", "vision_text")

# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def _img_to_b64(img_path: Path) -> tuple[str, str]:
    """Returns (mime_type, base64_string)."""
    ext  = img_path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    b64  = base64.b64encode(img_path.read_bytes()).decode()
    return mime, b64


def _parse_response(content: str) -> dict:
    """Extract JSON from model response, handling markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"misleading": None, "misleader_types": [], "explanation": content, "parse_error": True}


# ── RQ1 / RQ2: Misviz ────────────────────────────────────────────────────────

def _run_single_misviz(client: OpenAI, model: str, image_url: str,
                       condition: str, bbox_text: str = "") -> dict:
    prompt = (
        build_vision_only_prompt() if condition == "vision_only"
        else build_vision_text_prompt(bbox_text)
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            max_tokens=512,
        )
        if not response.choices:
            return {"misleading": None, "misleader_types": [], "explanation": "Empty response", "api_error": True}
        raw = response.choices[0].message.content
        if raw is None:
            return {"misleading": None, "misleader_types": [], "explanation": "Null content", "api_error": True}
        return _parse_response(raw)
    except Exception as e:
        return {"misleading": None, "misleader_types": [], "explanation": str(e), "api_error": True}


def evaluate(api_key: str, model_key: str, condition: str, n_samples: int = None) -> list[dict]:
    """
    Run RQ1/RQ2 evaluation on Misviz dataset.

    Args:
        api_key:    OpenRouter API key
        model_key:  'claude' or 'qwen'
        condition:  'vision_only' or 'vision_text'
        n_samples:  number of samples (None = full 2604)
    """
    assert model_key in MODELS,    f"model_key must be one of {list(MODELS)}"
    assert condition in CONDITIONS, f"condition must be one of {CONDITIONS}"

    model  = MODELS[model_key]
    client = _make_client(api_key)

    login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    dataset = load_dataset("UKPLab/misviz", split="train")
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    results = []
    correct, total = 0, 0

    print(f"Model: {model} | Condition: {condition} | Samples: {len(dataset)}")

    for i, item in enumerate(tqdm(dataset, desc=f"{model_key}/{condition}", unit="sample")):
        bboxes    = item.get("bbox", [])
        bbox_text = build_bbox_text(bboxes)

        buf = BytesIO()
        item["image"].save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        pred = _run_single_misviz(
            client=client,
            model=model,
            image_url=f"data:image/png;base64,{img_b64}",
            condition=condition,
            bbox_text=bbox_text,
        )

        gt_labels      = set(item.get("misleader", []))
        pred_labels    = set(pred.get("misleader_types", []))
        gt_binary      = len(gt_labels) > 0
        pred_binary    = pred.get("misleading", False)
        binary_correct = (gt_binary == pred_binary)

        if binary_correct:
            correct += 1
        total += 1

        results.append({
            "id":                   i,
            "gt_misleaders":        list(gt_labels),
            "chart_type":           item.get("chart_type", []),
            "bbox":                 bboxes,
            "pred_misleading":      pred_binary,
            "pred_misleader_types": list(pred_labels),
            "explanation":          pred.get("explanation", ""),
            "binary_correct":       binary_correct,
            "type_match":           (gt_labels == pred_labels),
            "parse_error":          pred.get("parse_error", False),
            "api_error":            pred.get("api_error", False),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(dataset)}] binary acc so far: {correct/total:.3f}")

        time.sleep(1.0)

    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{model_key}_{condition}.json"
    with open(fname, "w") as f:
        json.dump({
            "model":           model,
            "condition":       condition,
            "n_samples":       total,
            "binary_accuracy": correct / total if total else 0,
            "results":         results,
        }, f, indent=2)

    print(f"\nBinary accuracy: {correct/total:.3f} ({correct}/{total})")
    print(f"Saved → {fname}")
    return results


# ── RQ3: SEC filings ──────────────────────────────────────────────────────────

SKIP_KEYWORDS = [
    'logo', 'headshot', 'lineup', 'photo', 'portrait',
    'beer', 'wine', 'spirits', 'esg', 'map', 'facilities',
    'pipeline', 'terminal', 'co2', 'products', 'outlet',
    'newlands', 'bourdeau', 'hankinson', 'mcgrew',
    'monteiro', 'sabia', 'glaetzer', 'carey', 'hanson',
    'erickson', 'dykes', 'zeiler', 'walsh', 'khetani',
    '_g1',
]

# SEC 10-K stock performance comparison charts (cumulative return graphs)
# conventionally named _g2 — no Non-GAAP violation value, skip to save tokens
STOCK_RETURN_PATTERNS = ['_g2.']


def _is_financial_visual(item: dict) -> bool:
    """Keyword-based pre-filter: remove logos, headshots, product images, stock return charts."""
    alt   = item.get("alt", "").lower()
    fname = item.get("filename", "").lower()
    if any(kw in alt or kw in fname for kw in SKIP_KEYWORDS):
        return False
    if any(p in fname for p in STOCK_RETURN_PATTERNS):
        return False
    return True


def _is_financial_chart_by_vlm(client: OpenAI, model: str, img_path: Path) -> bool:
    """VLM pre-screen: confirm image is a financial chart/table."""
    mime, b64 = _img_to_b64(img_path)
    prompt = (
        "Is this image a financial chart or table (e.g., bar chart, line chart, "
        "pie chart, or data table showing financial metrics like revenue, EPS, "
        "margins, or growth)?\n\n"
        "Answer YES if it is a financial chart or table, NO otherwise.\n"
        "Your response must contain either YES or NO."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            max_tokens=16,  # slightly more room for Qwen
        )
        raw = response.choices[0].message.content or ""
        print(f"  [pre-screen] {img_path.name} → '{raw.strip()}'")
        return "YES" in raw.strip().upper()
    except Exception as e:
        print(f"  [pre-screen] ERROR: {e}")
        return False


def _build_sec_context(ticker: str, ground_truth: dict) -> str:
    """Build SEC comment letter context string for a ticker."""
    entries = ground_truth.get(ticker, [])
    uploads = [e for e in entries if e["form"] == "UPLOAD"]
    if not uploads:
        return "No SEC comment letter violations found for this company."
    lines = [f"SEC Comment Letter Violations for {ticker}:"]
    for entry in uploads[:3]:
        lines.append(f"\nDate: {entry['date']}")
        for m in entry["mentions"][:2]:
            lines.append(f"  - {m['anchor_sentence'][:300]}")
    return "\n".join(lines)


def _run_single_sec(client: OpenAI, model: str, img_path: Path, sec_context: str) -> dict:
    """Run VLM on a single SEC chart with optional context."""
    prompt     = build_rq3_prompt(sec_context)
    mime, b64  = _img_to_b64(img_path)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            max_tokens=512,
        )
        if not response.choices:
            return {"misleading": None, "sec_violation": None, "explanation": "Empty response", "api_error": True}
        raw = response.choices[0].message.content
        if raw is None:
            return {"misleading": None, "sec_violation": None, "explanation": "Null content", "api_error": True}

        content = raw.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"misleading": None, "sec_violation": None, "explanation": content, "parse_error": True}

    except Exception as e:
        return {"misleading": None, "sec_violation": None, "explanation": str(e), "api_error": True}


def evaluate_sec(api_key: str, model_key: str = "claude",
                 condition: str = "vision_text", max_per_ticker: int = None):
    """
    Run RQ3 evaluation on SEC 10-K visual presentations.
    Merges charts + tables manifests per ticker.

    Args:
        api_key:    OpenRouter API key
        model_key:  'claude' or 'qwen'
        condition:  'vision_text' or 'vision_only'
    """
    assert model_key in MODELS,    f"model_key must be one of {list(MODELS)}"
    assert condition in CONDITIONS, f"condition must be one of {CONDITIONS}"

    model  = MODELS[model_key]
    client = _make_client(api_key)

    # merge charts + tables manifests
    manifest = {}
    for visual_type in ("charts", "tables"):
        manifest_path = Path(f"data/{visual_type}/manifest.json")
        if not manifest_path.exists():
            print(f"⚠ Manifest not found: {manifest_path}, skipping")
            continue
        for ticker, items in json.loads(manifest_path.read_text()).items():
            manifest.setdefault(ticker, [])
            for item in items:
                item["visual_type"] = visual_type  # tag source
                manifest[ticker].append(item)

    ground_truth = json.loads(Path("data/ground_truth.json").read_text())

    results = {}
    total, flagged = 0, 0

    # count total filtered items upfront for progress bar
    all_items = [
        (ticker, item)
        for ticker, items in manifest.items()
        if items
        for item in items
        if _is_financial_visual(item)
    ]
    if max_per_ticker:
        from itertools import groupby
        all_items = [
            item for ticker, items in
            {t: [x for _, x in grp][:max_per_ticker]
             for t, grp in groupby(all_items, key=lambda x: x[0])}.items()
            for item in [(ticker, i) for i in items]
        ]

    pbar = tqdm(total=len(all_items), desc=f"{model_key}/{condition}", unit="img")

    for ticker, items in manifest.items():
        if not items:
            continue

        sec_context = (
            _build_sec_context(ticker, ground_truth)
            if condition == "vision_text"
            else ""
        )
        has_gt      = bool(ground_truth.get(ticker))
        results[ticker] = []

        items_filtered = [item for item in items if _is_financial_visual(item)]
        if max_per_ticker:
            items_filtered = items_filtered[:max_per_ticker]
        if not items_filtered:
            print(f"  ✗ {ticker}: no visuals after keyword filter")
            continue

        n_charts = sum(1 for i in items_filtered if i.get("visual_type") == "charts")
        n_tables = sum(1 for i in items_filtered if i.get("visual_type") == "tables")
        print(f"\n{'='*50}")
        print(f"{ticker} | charts={n_charts} tables={n_tables} | condition: {condition} | GT: {has_gt}")

        for item in tqdm(items_filtered, desc=f"{ticker}", unit="img", leave=False):
            img_path = Path(item["path"])
            if not img_path.exists():
                print(f"  ✗ Image not found: {img_path}")
                continue

            if not _is_financial_chart_by_vlm(client, model, img_path):
                print(f"  ⏭ Skipped (not financial): {img_path.name}")
                pbar.update(1)
                continue

            pred = _run_single_sec(client, model, img_path, sec_context)

            is_flagged = bool(pred.get("misleading") or pred.get("sec_violation"))
            if is_flagged:
                flagged += 1
            total += 1

            results[ticker].append({
                "file":             item.get("filename") or item.get("alt", ""),
                "date":             item.get("date", ""),
                "pred_misleading":  pred.get("misleading"),
                "pred_violation":   pred.get("sec_violation"),
                "explanation":      pred.get("explanation", ""),
                "has_gt_violation": has_gt,
                "parse_error":      pred.get("parse_error", False),
                "api_error":        pred.get("api_error", False),
            })

            pbar.set_postfix({"flagged": flagged, "total": total})
            pbar.update(1)
            print(f"  {'🚩' if is_flagged else '✓'} {item.get('filename', '')} "
                  f"→ {pred.get('sec_violation', 'None')}")
            time.sleep(1.0)

    pbar.close()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"sec_{model_key}_{condition}.json"
    with open(fname, "w") as f:
        json.dump({
            "model":     model,
            "condition": condition,
            "total":     total,
            "flagged":   flagged,
            "flag_rate": flagged / total if total else 0,
            "results":   results,
        }, f, indent=2)

    print(f"\n📊 Flagged: {flagged}/{total} ({flagged/total:.1%})" if total else "\n📊 No items evaluated")
    print(f"💾 Saved → {fname}")
    return results