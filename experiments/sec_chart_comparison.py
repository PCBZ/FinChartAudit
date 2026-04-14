"""SEC Chart comparison: VLM-only vs LLM-OCR+Rules on our extracted SEC charts.

Tests whether tool augmentation helps on real SEC filing charts
(vs Misviz benchmark charts).

Usage:
    python run_sec_chart_comparison.py
"""
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from finchartaudit.config import get_config
from finchartaudit.tools.rule_check import RuleEngine
from finchartaudit.prompts.t2_visual import MISLEADER_DEFINITIONS, COMPLETENESS_CHECKS
from finchartaudit.agents.t2_pipeline import PIPELINE_SYSTEM_PROMPT, PIPELINE_PROMPT

OUT_DIR = Path("data/eval_results/sec_chart_comparison")
CHARTS_DIR = Path("data/charts")

# ── Prompts ──────────────────────────────────────────────────────────────────

VLM_ONLY_PROMPT = """You are an expert in financial data visualization and SEC compliance. Analyze this chart from an SEC filing for misleading visual elements.

## Misleader Taxonomy
- misrepresentation: bar/area sizes do not match labeled values
- 3d: 3D effects distort visual comparison
- truncated axis: y-axis doesn't start at zero, exaggerating differences
- inappropriate use of pie chart: used for data unsuitable for part-to-whole comparison
- inconsistent tick intervals: axis ticks are unevenly spaced
- dual axis: two y-axes with different scales mislead comparisons
- inconsistent binning size: histogram bins have unequal widths without normalization
- discretized continuous variable: continuous data binned to hide distribution
- inappropriate use of line chart: used for non-sequential or categorical data
- inappropriate item order: ordering creates false impressions
- inverted axis: axis direction reversed, flipping perceived trend
- inappropriate axis range: range set to exaggerate or minimize differences

## Output
Respond with valid JSON only:
{
  "misleading": true/false,
  "misleader_types": ["zero or more types from the taxonomy"],
  "explanation": "one to three sentences describing what you found"
}"""

OCR_SYSTEM = "You are an OCR tool. You ONLY read and transcribe visible text from images. Do NOT interpret, analyze, or judge the content."

OCR_PROMPT = """Read ALL visible text in this chart image. Transcribe exactly what you see, organized by region.

Return ONLY valid JSON:
{
  "chart_type": "bar|line|pie|area|scatter|histogram|other",
  "title": "exact title text or empty string",
  "y_axis_label": "exact Y-axis label text or empty string",
  "y_axis_values": ["list", "of", "all", "Y-axis", "tick", "labels", "top", "to", "bottom"],
  "x_axis_label": "exact X-axis label text or empty string",
  "x_axis_values": ["list", "of", "all", "X-axis", "tick", "labels", "left", "to", "right"],
  "right_y_axis_label": "right Y-axis label or empty string if none",
  "right_y_axis_values": ["list of right Y-axis ticks, or empty if none"],
  "legend_items": ["list", "of", "legend", "labels"],
  "data_labels": ["any", "numbers", "or", "text", "directly", "on", "bars/points/slices"],
  "source_text": "any source attribution text or empty string",
  "other_text": ["any", "other", "visible", "text", "not", "covered", "above"]
}

Rules:
- Transcribe EXACTLY what you see - do not infer, round, or modify values
- If a region has no text, use empty string or empty list
- Do NOT add any analysis or commentary"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def img_to_b64(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode in ("CMYK", "RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def vlm_call(client, model, image_b64, system, prompt):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=2048, temperature=0.0)
    return response.choices[0].message.content or ""


def extract_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for marker in ["```json", "```"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def parse_numbers(values):
    numbers = []
    for v in values:
        if isinstance(v, (int, float)):
            numbers.append(float(v))
            continue
        if not isinstance(v, str):
            continue
        cleaned = re.sub(r'[,$%]', '', v.strip())
        cleaned = re.sub(r'[KkMmBb]$', '', cleaned)
        try:
            numbers.append(float(cleaned))
        except ValueError:
            pass
    return numbers


def run_rule_checks(engine, ocr_data):
    results = []
    y_values = parse_numbers(ocr_data.get("y_axis_values", []))
    x_values = parse_numbers(ocr_data.get("x_axis_values", []))
    right_values = parse_numbers(ocr_data.get("right_y_axis_values", []))
    chart_type = ocr_data.get("chart_type", "unknown")

    if y_values:
        for check, key in [("truncated_axis", "is_truncated"), ("broken_scale", "is_broken"),
                           ("inverted_axis", "is_inverted")]:
            try:
                r = engine.run_check(check, {"axis_values": y_values, "chart_type": chart_type})
                if r.get(key):
                    results.append(f"{check}: {r['explanation']}")
            except Exception:
                pass
        try:
            r = engine.run_check("inappropriate_axis_range", {"axis_values": y_values, "chart_type": chart_type})
            if r["is_inappropriate"]:
                results.append(f"inappropriate_axis_range: {r['explanation']}")
        except Exception:
            pass

    if y_values and right_values:
        try:
            r = engine.run_check("dual_axis", {"left_axis_values": y_values, "right_axis_values": right_values})
            if r["has_dual_axis"]:
                results.append(f"dual_axis: {r['explanation']}")
        except Exception:
            pass

    if len(x_values) >= 3:
        try:
            r = engine.run_check("inconsistent_binning", {"bin_edges": x_values})
            if r["is_inconsistent"]:
                results.append(f"inconsistent_binning: {r['explanation']}")
        except Exception:
            pass

    return results


def build_ocr_summary(ocr_data):
    lines = []
    for key, label in [("title", "Title"), ("y_axis_label", "Y-axis label"),
                       ("y_axis_values", "Y-axis values"), ("x_axis_label", "X-axis label"),
                       ("x_axis_values", "X-axis values"), ("right_y_axis_label", "Right Y-axis"),
                       ("right_y_axis_values", "Right Y-axis values"),
                       ("legend_items", "Legend"), ("data_labels", "Data labels"),
                       ("source_text", "Source")]:
        val = ocr_data.get(key)
        if val:
            if isinstance(val, list) and val:
                lines.append(f"{label}: {', '.join(str(v) for v in val)}")
            elif isinstance(val, str) and val:
                lines.append(f"{label}: {val}")
    return "\n".join(lines) if lines else "No text detected."


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    get_config.cache_clear()
    config = get_config()
    model = config.vlm_model
    print(f"Model: {model}")

    from openai import OpenAI
    client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)
    engine = RuleEngine()

    # Collect all chart images
    charts = []
    for ticker_dir in sorted(CHARTS_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for img_path in sorted(ticker_dir.glob("*")):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                charts.append({
                    "ticker": ticker_dir.name,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                })

    print(f"Total SEC charts: {len(charts)}")

    results_vlm = []
    results_ocr = []
    lock = threading.Lock()
    completed = 0

    def process_one(chart):
        nonlocal completed
        ticker = chart["ticker"]
        img_name = chart["image_name"]
        image_b64 = img_to_b64(chart["image_path"])

        # Condition 1: VLM-only
        try:
            t1 = time.time()
            raw = vlm_call(client, model, image_b64, "", VLM_ONLY_PROMPT)
            elapsed1 = time.time() - t1
            parsed = extract_json(raw) or {}
            r1 = {
                "ticker": ticker, "image": img_name,
                "misleading": parsed.get("misleading", False),
                "misleader_types": parsed.get("misleader_types", []),
                "explanation": parsed.get("explanation", ""),
                "elapsed_s": round(elapsed1, 1),
            }
        except Exception as e:
            r1 = {"ticker": ticker, "image": img_name, "error": str(e)}

        # Condition 2: LLM-OCR + Rules
        try:
            t2 = time.time()
            raw_ocr = vlm_call(client, model, image_b64, OCR_SYSTEM, OCR_PROMPT)
            ocr_data = extract_json(raw_ocr) or {}
            rule_results = run_rule_checks(engine, ocr_data)
            ocr_summary = build_ocr_summary(ocr_data)
            y_values = parse_numbers(ocr_data.get("y_axis_values", []))

            misleader_list = "\n".join(f"- {k}: {v}" for k, v in MISLEADER_DEFINITIONS.items())
            completeness_list = "\n".join(f"- {k}: {v}" for k, v in COMPLETENESS_CHECKS.items())
            judge_prompt = PIPELINE_PROMPT.format(
                chart_id=f"{ticker}_{img_name}", page=1,
                ocr_text=ocr_summary,
                ocr_axis=", ".join(str(v) for v in y_values) if y_values else "No axis values.",
                rule_results="\n".join(rule_results) if rule_results else "No rule checks applicable.",
                misleader_list=misleader_list, completeness_list=completeness_list,
            )
            raw_judge = vlm_call(client, model, image_b64, PIPELINE_SYSTEM_PROMPT, judge_prompt)
            elapsed2 = time.time() - t2
            judge_data = extract_json(raw_judge) or {}

            misleaders = []
            for name, a in judge_data.get("misleaders", {}).items():
                if isinstance(a, dict) and a.get("present") and float(a.get("confidence", 0)) >= 0.3:
                    misleaders.append(name)

            completeness = []
            for name, a in judge_data.get("completeness", {}).items():
                if isinstance(a, dict) and a.get("present") and float(a.get("confidence", 0)) >= 0.3:
                    completeness.append(name)

            r2 = {
                "ticker": ticker, "image": img_name,
                "misleaders": list(set(misleaders)),
                "completeness": completeness,
                "rule_results": rule_results,
                "chart_type": ocr_data.get("chart_type", ""),
                "elapsed_s": round(elapsed2, 1),
            }
        except Exception as e:
            r2 = {"ticker": ticker, "image": img_name, "error": str(e)}

        with lock:
            completed += 1
            results_vlm.append(r1)
            results_ocr.append(r2)
            flag1 = r1.get("misleading", False)
            types1 = r1.get("misleader_types", [])
            types2 = r2.get("misleaders", [])
            rules = len(r2.get("rule_results", []))
            compl = r2.get("completeness", [])
            print(f"[{completed}/{len(charts)}] {ticker}/{img_name}")
            print(f"  VLM-only: flag={flag1} types={types1}")
            print(f"  LLM-OCR+Rules: misleaders={types2} completeness={len(compl)} rules={rules}")

    print(f"\nRunning 2-condition comparison on {len(charts)} SEC charts...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_one, c) for c in charts]
        for f in as_completed(futures):
            try:
                f.result(timeout=180)
            except Exception as e:
                print(f"Thread error: {e}")

    total_time = time.time() - t0

    # Save
    (OUT_DIR / "vlm_only.json").write_text(json.dumps(results_vlm, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "llm_ocr_rules.json").write_text(json.dumps(results_ocr, indent=2, default=str), encoding="utf-8")

    # Summary
    print(f"\n{'='*60}")
    print("SEC CHART COMPARISON SUMMARY")
    print(f"{'='*60}")

    vlm_flagged = sum(1 for r in results_vlm if r.get("misleading") and "error" not in r)
    vlm_total = sum(1 for r in results_vlm if "error" not in r)
    ocr_flagged = sum(1 for r in results_ocr if r.get("misleaders") and "error" not in r)
    ocr_total = sum(1 for r in results_ocr if "error" not in r)

    print(f"VLM-only: {vlm_flagged}/{vlm_total} flagged ({vlm_flagged/max(vlm_total,1)*100:.0f}%)")
    print(f"LLM-OCR+Rules: {ocr_flagged}/{ocr_total} flagged ({ocr_flagged/max(ocr_total,1)*100:.0f}%)")

    # Per-ticker
    from collections import defaultdict
    ticker_vlm = defaultdict(lambda: {"flagged": 0, "total": 0})
    ticker_ocr = defaultdict(lambda: {"flagged": 0, "total": 0})
    for r in results_vlm:
        if "error" not in r:
            ticker_vlm[r["ticker"]]["total"] += 1
            if r.get("misleading"): ticker_vlm[r["ticker"]]["flagged"] += 1
    for r in results_ocr:
        if "error" not in r:
            ticker_ocr[r["ticker"]]["total"] += 1
            if r.get("misleaders"): ticker_ocr[r["ticker"]]["flagged"] += 1

    print(f"\n{'Ticker':<8} {'Charts':>7} {'VLM Flag':>9} {'OCR+R Flag':>11}")
    print("-" * 38)
    for ticker in sorted(set(list(ticker_vlm.keys()) + list(ticker_ocr.keys()))):
        v = ticker_vlm[ticker]
        o = ticker_ocr[ticker]
        print(f"{ticker:<8} {v['total']:>7} {v['flagged']:>9} {o['flagged']:>11}")

    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
