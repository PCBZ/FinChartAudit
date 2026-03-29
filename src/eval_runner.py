# src/eval_runner.py

import json
import os
import base64
import threading
from pathlib import Path
from io import BytesIO
from PIL import Image
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts import (
    build_vision_only_prompt,
    build_vision_text_prompt,
    build_bbox_text,
    build_rq3_prompt,
    MISLEADER_TYPES,
)

# ── Model configurations ──────────────────────────────────────────────────────

MODELS = {
    "claude": "anthropic/claude-haiku-4.5",
    "qwen":   "qwen/qwen3-vl-8b-instruct",
}

CONDITIONS = ("vision_only", "vision_text")

# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def _img_to_b64(img_path: Path, max_bytes: int = 4 * 1024 * 1024) -> tuple[str, str]:
    """Returns (mime_type, base64_string), compressing to JPEG if needed."""
    img = Image.open(img_path)
    if img.mode in ("CMYK", "RGBA", "P"):
        img = img.convert("RGB")
    quality = 85
    while True:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes or quality <= 30:
            break
        quality -= 10
    buf.seek(0)
    return "image/jpeg", base64.b64encode(buf.read()).decode()


def _parse_response(content: str) -> dict:
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _call_misviz_api(client: OpenAI, model: str, image_url: str,
                     condition: str, bbox_text: str = "") -> str:
    """Raw API call — raises on error so tenacity can retry."""
    prompt = (
        build_vision_only_prompt() if condition == "vision_only"
        else build_vision_text_prompt(bbox_text)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
        max_tokens=512,
        timeout=30,
    )
    if not response.choices or response.choices[0].message.content is None:
        raise ValueError("Empty response")
    return response.choices[0].message.content


def _run_single_misviz(client: OpenAI, model: str, image_url: str,
                       condition: str, bbox_text: str = "") -> dict:
    try:
        raw = _call_misviz_api(client, model, image_url, condition, bbox_text)
        return _parse_response(raw)
    except Exception as e:
        return {"misleading": None, "misleader_types": [], "explanation": str(e), "api_error": True}


def _stratified_sample(dataset, n_per_type: int = 15, n_clean: int = 100) -> list:
    """Stratified sample: n_per_type per misleader type + n_clean clean samples."""
    type_buckets = defaultdict(list)
    clean_bucket = []
    for i, item in enumerate(dataset):
        misleaders = item.get("misleader", [])
        if not misleaders:
            clean_bucket.append(i)
        else:
            for t in misleaders:
                type_buckets[t].append(i)
    selected = set()
    for t, indices in type_buckets.items():
        selected.update(indices[:n_per_type])
    selected.update(clean_bucket[:n_clean])
    return sorted(selected)


def evaluate(api_key: str, model_key: str, condition: str,
             n_samples: int = None, n_per_type: int = 15, n_clean: int = 100,
             workers: int = 8) -> list[dict]:
    """Run RQ1/RQ2 evaluation on Misviz dataset."""
    assert model_key in MODELS,    f"model_key must be one of {list(MODELS)}"
    assert condition in CONDITIONS, f"condition must be one of {CONDITIONS}"

    model  = MODELS[model_key]
    client = _make_client(api_key)

    login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    ds      = load_dataset("UKPLab/misviz")
    dataset = ds["test"]
    print(f"Dataset size: {len(dataset)}")

    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    else:
        indices = _stratified_sample(dataset, n_per_type=n_per_type, n_clean=n_clean)
        dataset = dataset.select(indices)
        print(f"Stratified sample: {len(dataset)} samples")

    results        = []
    correct, total = 0, 0
    lock           = threading.Lock()

    print(f"Model: {model} | Condition: {condition} | Samples: {len(dataset)}")

    def process_item(args):
        i, item   = args
        bboxes    = item.get("bbox", [])
        bbox_text = build_bbox_text(bboxes)

        img = item["image"]
        if img.mode in ("CMYK", "RGBA", "P"):
            img = img.convert("RGB")
        quality = 85
        while True:
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            if buf.tell() <= 4 * 1024 * 1024 or quality <= 30:
                break
            quality -= 10
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()

        pred = _run_single_misviz(
            client=client,
            model=model,
            image_url=f"data:image/jpeg;base64,{img_b64}",
            condition=condition,
            bbox_text=bbox_text,
        )

        gt_labels      = set(item.get("misleader", []))
        pred_labels    = set(pred.get("misleader_types", []))
        gt_binary      = len(gt_labels) > 0
        pred_binary    = pred.get("misleading", False)
        binary_correct = (gt_binary == pred_binary)

        return {
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
        }

    pbar = tqdm(total=len(dataset), desc=f"{model_key}/{condition}", unit="sample")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_item, (i, item)): i
                   for i, item in enumerate(dataset)}
        for future in as_completed(futures):
            try:
                result = future.result(timeout=60)
            except Exception as e:
                print(f"\n[TIMEOUT/ERROR] {e}")
                result = None
            with lock:
                if result is None:
                    pbar.update(1)
                    continue
                results.append(result)
                if result["binary_correct"]:
                    correct += 1
                total += 1
                pbar.update(1)
                pbar.set_postfix({"acc": f"{correct/total:.3f}"})

    pbar.close()
    results.sort(key=lambda x: x["id"])

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

STOCK_RETURN_PATTERNS = ['_g2.']


def _is_financial_visual(item: dict) -> bool:
    alt   = item.get("alt", "").lower()
    fname = item.get("filename", "").lower()
    if any(kw in alt or kw in fname for kw in SKIP_KEYWORDS):
        return False
    if any(p in fname for p in STOCK_RETURN_PATTERNS):
        return False
    return True


def _load_prescreen_cache() -> dict:
    cache_path = Path("data/prescreen_cache.json")
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_prescreen_cache(cache: dict):
    cache_path = Path("data/prescreen_cache.json")
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _is_financial_chart_by_vlm(client: OpenAI, model: str, img_path: Path,
                                cache: dict) -> bool:
    cache_key = str(img_path)
    if cache_key in cache:
        return cache[cache_key]

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
            max_tokens=16,
        )
        raw    = response.choices[0].message.content or ""
        result = "YES" in raw.strip().upper()
        cache[cache_key] = result
        return result
    except Exception as e:
        print(f"  [pre-screen] ERROR: {e}")
        return False


def _build_sec_context(ticker: str, ground_truth: dict) -> str:
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _call_sec_api(client: OpenAI, model: str, img_path: Path, sec_context: str) -> str:
    """Raw API call — raises on error so tenacity can retry."""
    prompt    = build_rq3_prompt(sec_context)
    mime, b64 = _img_to_b64(img_path)
    response  = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
        max_tokens=512,
        timeout=30,
    )
    if not response.choices or response.choices[0].message.content is None:
        raise ValueError("Empty response")
    return response.choices[0].message.content


def _run_single_sec(client: OpenAI, model: str, img_path: Path, sec_context: str) -> dict:
    try:
        raw = _call_sec_api(client, model, img_path, sec_context)
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
                 condition: str = "vision_text", max_per_ticker: int = 10,
                 workers: int = 8):
    """Run RQ3 evaluation on SEC 10-K visual presentations."""
    assert model_key in MODELS,    f"model_key must be one of {list(MODELS)}"
    assert condition in CONDITIONS, f"condition must be one of {CONDITIONS}"

    model  = MODELS[model_key]
    client = _make_client(api_key)

    manifest = {}
    for visual_type in ("charts", "tables"):
        manifest_path = Path(f"data/{visual_type}/manifest.json")
        if not manifest_path.exists():
            print(f"⚠ Manifest not found: {manifest_path}, skipping")
            continue
        for ticker, items in json.loads(manifest_path.read_text()).items():
            manifest.setdefault(ticker, [])
            for item in items:
                item["visual_type"] = visual_type
                manifest[ticker].append(item)

    ground_truth    = json.loads(Path("data/ground_truth.json").read_text())
    prescreen_cache = _load_prescreen_cache()

    results    = {}
    total      = 0
    flagged    = 0
    lock       = threading.Lock()
    cache_lock = threading.Lock()

    all_items = [
        (ticker, item)
        for ticker, items in manifest.items()
        if items and ground_truth.get(ticker)  # only GT tickers
        for item in items
        if _is_financial_visual(item)
    ]
    if max_per_ticker:
        from itertools import groupby
        all_items = [
            (ticker, item)
            for ticker, grp in {
                t: [x for _, x in g][:max_per_ticker]
                for t, g in groupby(all_items, key=lambda x: x[0])
            }.items()
            for item in grp
        ]

    pbar = tqdm(total=len(all_items), desc=f"{model_key}/{condition}", unit="img")

    for ticker, items in manifest.items():
        if not items:
            continue

        has_gt = bool(ground_truth.get(ticker))
        if not has_gt:
            print(f"  ⏭ {ticker}: no GT violations, skipping")
            continue

        sec_context     = _build_sec_context(ticker, ground_truth) if condition == "vision_text" else ""
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

        def process_item(item):
            img_path = Path(item["path"])
            if not img_path.exists():
                return None

            with cache_lock:
                cached = prescreen_cache.get(str(img_path))
            if cached is None:
                result = _is_financial_chart_by_vlm(client, model, img_path, prescreen_cache)
                with cache_lock:
                    prescreen_cache[str(img_path)] = result
            else:
                result = cached

            if not result:
                pbar.update(1)
                return None

            pred       = _run_single_sec(client, model, img_path, sec_context)
            is_flagged = bool(pred.get("misleading") or pred.get("sec_violation"))

            return {
                "file":             item.get("filename") or item.get("alt", ""),
                "date":             item.get("date", ""),
                "pred_misleading":  pred.get("misleading"),
                "pred_violation":   pred.get("sec_violation"),
                "explanation":      pred.get("explanation", ""),
                "has_gt_violation": has_gt,
                "parse_error":      pred.get("parse_error", False),
                "api_error":        pred.get("api_error", False),
                "is_flagged":       is_flagged,
            }

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_item, item) for item in items_filtered]
            for future in as_completed(futures):
                try:
                    res = future.result(timeout=60)
                except Exception as e:
                    print(f"\n[TIMEOUT/ERROR] {e}")
                    pbar.update(1)
                    continue
                pbar.update(1)
                if res is None:
                    continue
                with lock:
                    if res["is_flagged"]:
                        flagged += 1
                    total += 1
                    pbar.set_postfix({"flagged": flagged, "total": total})
                    results[ticker].append({k: v for k, v in res.items() if k != "is_flagged"})
                    print(f"  {'🚩' if res['is_flagged'] else '✓'} {res['file']} "
                          f"→ {res['pred_violation'] or 'None'}")

    pbar.close()
    _save_prescreen_cache(prescreen_cache)

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