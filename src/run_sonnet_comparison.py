"""Sonnet comparison: VLM-only vs LLM-OCR+Rules on 50 charts.

Tests whether a larger model (Sonnet) can benefit from tool augmentation,
unlike Haiku which showed no benefit.

Usage:
    python run_sonnet_comparison.py
"""
import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from finchartaudit.tools.rule_check import RuleEngine
from finchartaudit.prompts.t2_visual import MISLEADER_DEFINITIONS, COMPLETENESS_CHECKS
from finchartaudit.agents.t2_pipeline import PIPELINE_SYSTEM_PROMPT, PIPELINE_PROMPT

OUT_DIR = Path("data/eval_results/sonnet_comparison")
SONNET_MODEL = "anthropic/claude-sonnet-4"

# Reuse prompts from run_llm_ocr_rules.py
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
- For axis values, preserve the exact format (e.g., "$1.2M", "50%", "2024Q1")
- Read Y-axis top to bottom, X-axis left to right
- If a region has no text, use empty string or empty list
- Do NOT add any analysis or commentary"""

# B's VLM-only prompt (simplified)
VLM_ONLY_PROMPT = """You are an expert in data visualization. Detect misleading elements in the chart image.

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
  "explanation": "one to three sentences"
}"""


def select_50_stratified():
    """Select 50 samples stratified from B's 271."""
    from data_tools.misviz.loader import MisvizLoader

    b_path = Path("C:/Users/chntw/Documents/7180/PCBZ_FinChartAudit/results/claude_vision_only.json")
    b_data = json.loads(b_path.read_text(encoding="utf-8"))
    b_items = b_data["results"]

    loader = MisvizLoader()
    real_data = loader.load_real()

    content_to_local = defaultdict(list)
    for i, d in enumerate(real_data):
        key = (frozenset(d.get("misleader", [])), tuple(sorted(d.get("chart_type", []))))
        content_to_local[key].append(i)

    # Match all 271, then subsample 50
    all_samples = []
    used = set()
    for b_item in b_items:
        key = (frozenset(b_item["gt_misleaders"]), tuple(sorted(b_item.get("chart_type", []))))
        candidates = [idx for idx in content_to_local.get(key, []) if idx not in used]
        if candidates:
            idx = candidates[0]
            used.add(idx)
            instance = loader.get_real_instance(idx)
            if Path(instance.image_path).exists():
                all_samples.append({
                    "idx": idx,
                    "instance_id": str(b_item["id"]),
                    "image_path": instance.image_path,
                    "ground_truth": instance.misleader,
                })

    # Stratified: 3 per type + 14 clean = ~50
    type_buckets = defaultdict(list)
    clean = []
    for s in all_samples:
        if not s["ground_truth"]:
            clean.append(s)
        else:
            for t in s["ground_truth"]:
                type_buckets[t].append(s)

    selected_ids = set()
    selected = []
    for t, items in type_buckets.items():
        for item in items:
            if item["instance_id"] not in selected_ids and len([s for s in selected if t in s["ground_truth"]]) < 3:
                selected.append(item)
                selected_ids.add(item["instance_id"])

    for item in clean:
        if len(selected) >= 50:
            break
        if item["instance_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["instance_id"])

    print(f"Selected {len(selected)} samples for Sonnet comparison")
    return selected[:50]


def img_to_b64(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode in ("CMYK", "RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def vlm_call(client, model, image_b64, system, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]},
        ],
        max_tokens=2048,
        temperature=0.0,
    )
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
        try:
            r = engine.run_check("truncated_axis", {"axis_values": y_values, "chart_type": chart_type})
            if r["is_truncated"]:
                results.append(f"truncated_axis: {r['explanation']}")
        except Exception: pass
        try:
            r = engine.run_check("broken_scale", {"axis_values": y_values})
            if r["is_broken"]:
                results.append(f"broken_scale: {r['explanation']}")
        except Exception: pass
        try:
            r = engine.run_check("inverted_axis", {"axis_values": y_values})
            if r["is_inverted"]:
                results.append(f"inverted_axis: {r['explanation']}")
        except Exception: pass
        try:
            r = engine.run_check("inappropriate_axis_range", {"axis_values": y_values, "chart_type": chart_type})
            if r["is_inappropriate"]:
                results.append(f"inappropriate_axis_range: {r['explanation']}")
        except Exception: pass

    if y_values and right_values:
        try:
            r = engine.run_check("dual_axis", {"left_axis_values": y_values, "right_axis_values": right_values})
            if r["has_dual_axis"]:
                results.append(f"dual_axis: {r['explanation']}")
        except Exception: pass

    if len(x_values) >= 3:
        try:
            r = engine.run_check("inconsistent_binning", {"bin_edges": x_values})
            if r["is_inconsistent"]:
                results.append(f"inconsistent_binning: {r['explanation']}")
        except Exception: pass

    return results


def build_ocr_summary(ocr_data):
    lines = []
    if ocr_data.get("title"): lines.append(f"Title: {ocr_data['title']}")
    if ocr_data.get("y_axis_label"): lines.append(f"Y-axis label: {ocr_data['y_axis_label']}")
    if ocr_data.get("y_axis_values"): lines.append(f"Y-axis values: {', '.join(str(v) for v in ocr_data['y_axis_values'])}")
    if ocr_data.get("x_axis_label"): lines.append(f"X-axis label: {ocr_data['x_axis_label']}")
    if ocr_data.get("x_axis_values"): lines.append(f"X-axis values: {', '.join(str(v) for v in ocr_data['x_axis_values'])}")
    if ocr_data.get("right_y_axis_label"): lines.append(f"Right Y-axis: {ocr_data['right_y_axis_label']}")
    if ocr_data.get("right_y_axis_values"): lines.append(f"Right Y-axis values: {', '.join(str(v) for v in ocr_data['right_y_axis_values'])}")
    if ocr_data.get("legend_items"): lines.append(f"Legend: {', '.join(ocr_data['legend_items'])}")
    if ocr_data.get("data_labels"): lines.append(f"Data labels: {', '.join(str(v) for v in ocr_data['data_labels'])}")
    if ocr_data.get("source_text"): lines.append(f"Source: {ocr_data['source_text']}")
    return "\n".join(lines) if lines else "No text detected."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)  # Sonnet rate limits are tighter
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from finchartaudit.config import get_config
    get_config.cache_clear()
    config = get_config()

    from openai import OpenAI
    client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)
    engine = RuleEngine()

    samples = select_50_stratified()

    results_vlm_only = []
    results_llm_ocr_rules = []
    lock = threading.Lock()
    completed = 0

    def process_one(sample):
        nonlocal completed
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])

        # Condition 1: Sonnet VLM-only
        try:
            start = time.time()
            raw = vlm_call(client, SONNET_MODEL, image_b64, "", VLM_ONLY_PROMPT)
            t1 = time.time() - start
            parsed = extract_json(raw) or {}
            pred_vlm = parsed.get("misleader_types", [])
            r1 = {"instance_id": iid, "ground_truth": sample["ground_truth"],
                   "predicted": pred_vlm, "elapsed_s": round(t1, 1)}
        except Exception as e:
            r1 = {"instance_id": iid, "ground_truth": sample["ground_truth"],
                   "predicted": [], "error": str(e)}

        # Condition 2: Sonnet LLM-OCR + Rules
        try:
            start = time.time()
            raw_ocr = vlm_call(client, SONNET_MODEL, image_b64, OCR_SYSTEM, OCR_PROMPT)
            ocr_data = extract_json(raw_ocr) or {}
            rule_results = run_rule_checks(engine, ocr_data)
            ocr_summary = build_ocr_summary(ocr_data)
            y_values = parse_numbers(ocr_data.get("y_axis_values", []))
            ocr_axis_str = ", ".join(str(v) for v in y_values) if y_values else "No axis values."

            misleader_list = "\n".join(f"- {k}: {v}" for k, v in MISLEADER_DEFINITIONS.items())
            completeness_list = "\n".join(f"- {k}: {v}" for k, v in COMPLETENESS_CHECKS.items())
            judge_prompt = PIPELINE_PROMPT.format(
                chart_id=f"eval_{iid}", page=1,
                ocr_text=ocr_summary, ocr_axis=ocr_axis_str,
                rule_results="\n".join(rule_results) if rule_results else "No rule checks applicable.",
                misleader_list=misleader_list, completeness_list=completeness_list,
            )
            raw_judge = vlm_call(client, SONNET_MODEL, image_b64, PIPELINE_SYSTEM_PROMPT, judge_prompt)
            t2 = time.time() - start
            judge_data = extract_json(raw_judge) or {}
            pred_ocr = [name for name, a in judge_data.get("misleaders", {}).items()
                        if isinstance(a, dict) and a.get("present") and float(a.get("confidence", 0)) >= 0.3]
            r2 = {"instance_id": iid, "ground_truth": sample["ground_truth"],
                   "predicted": list(set(pred_ocr)), "elapsed_s": round(t2, 1),
                   "rule_results": rule_results}
        except Exception as e:
            r2 = {"instance_id": iid, "ground_truth": sample["ground_truth"],
                   "predicted": [], "error": str(e)}

        with lock:
            completed += 1
            results_vlm_only.append(r1)
            results_llm_ocr_rules.append(r2)
            gt = sample["ground_truth"]
            p1 = r1.get("predicted", [])
            p2 = r2.get("predicted", [])
            print(f"[{completed}/{len(samples)}] id={iid} gt={gt}")
            print(f"  VLM-only: {p1} ({r1.get('elapsed_s','ERR')}s)")
            print(f"  LLM-OCR+Rules: {p2} ({r2.get('elapsed_s','ERR')}s) rules={len(r2.get('rule_results',[]))}")

    print(f"Running Sonnet comparison: {len(samples)} charts x 2 conditions, {args.workers} threads")
    print(f"Model: {SONNET_MODEL}")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, s) for s in samples]
        for f in as_completed(futures):
            try:
                f.result(timeout=180)
            except Exception as e:
                print(f"Thread error: {e}")

    total_time = time.time() - t0

    # Save results
    (OUT_DIR / "vlm_only.json").write_text(json.dumps(results_vlm_only, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "llm_ocr_rules.json").write_text(json.dumps(results_llm_ocr_rules, indent=2, default=str), encoding="utf-8")

    # Compute metrics for both
    from data_tools.misviz.evaluator import MisvizEvaluator

    print(f"\n{'='*60}")
    print(f"SONNET VLM-ONLY")
    print(f"{'='*60}")
    ev1 = MisvizEvaluator()
    for r in results_vlm_only:
        if "error" not in r:
            ev1.add_prediction(r["instance_id"], r["ground_truth"], r["predicted"],
                              condition="sonnet_vlm_only", model="sonnet")
    ev1.print_summary()
    m1 = ev1.compute_metrics()
    (OUT_DIR / "metrics_vlm_only.json").write_text(json.dumps(m1, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"SONNET LLM-OCR+RULES")
    print(f"{'='*60}")
    ev2 = MisvizEvaluator()
    for r in results_llm_ocr_rules:
        if "error" not in r:
            ev2.add_prediction(r["instance_id"], r["ground_truth"], r["predicted"],
                              condition="sonnet_llm_ocr_rules", model="sonnet")
    ev2.print_summary()
    m2 = ev2.compute_metrics()
    (OUT_DIR / "metrics_llm_ocr_rules.json").write_text(json.dumps(m2, indent=2), encoding="utf-8")

    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
