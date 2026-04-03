"""V7 pipeline: B's prompt + sequential per-type re-ask for clean charts + CLEAN veto + DePlot axis_range.

Key insight: VLM CAN detect blind spot types when asked individually.
The general 12-type prompt dilutes attention. Solution: for charts Call 1
reports as clean, ask each blind spot type ONE AT A TIME.

Architecture:
  Call 1: General detection (all types, single prompt)
  Calls 2-N: ONLY for Call 1 clean charts → one call per blind spot type
  Post: OCR CLEAN veto + DePlot axis_range

Usage:
    python run_pipeline_v7_sequential.py --workers 8
"""
import argparse
import base64
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

OUT_DIR = Path("data/eval_results/v7_sequential")
OCR_CACHE_DIR = Path("data/eval_results/pipeline_full/ocr_cache")

# ── Call 1: same as V6a ────────────────────────────────────────────────────

TAXONOMY_BLOCK = """
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
""".strip()

FEW_SHOT_EXAMPLES = """
EXAMPLE 1
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
Chart: A 3D bar chart showing quarterly profits. The 3D perspective makes front bars appear larger than back bars.
Output:
{"misleading": true, "misleader_types": ["3d"], "explanation": "3D perspective distorts visual comparison of bar sizes. This is a 3D rendering issue, not misrepresentation."}

EXAMPLE 5
Chart: A line chart showing stock price from $45 to $52 over 6 months. The y-axis starts at $44.
Output:
{"misleading": false, "misleader_types": [], "explanation": "Line charts commonly use non-zero y-axis baselines to show trends clearly."}

EXAMPLE 6
Chart: A bar chart showing satisfaction scores from 4.1 to 4.5 on a scale of 1-5. The y-axis starts at 4.0.
Output:
{"misleading": true, "misleader_types": ["truncated axis"], "explanation": "The bar chart y-axis starts at 4.0 instead of 0, making small differences appear much larger."}
""".strip()

OUTPUT_FORMAT = '{"misleading": <true|false>, "misleader_types": [...], "explanation": "..."}'

CALL1_PROMPT = f"""You are an expert in data visualization. Detect misleading elements in the chart image.

## Misleader Taxonomy
{TAXONOMY_BLOCK}

## Examples
{FEW_SHOT_EXAMPLES}

## Output
Respond with valid JSON only:
{OUTPUT_FORMAT}"""

# ── Per-type targeted prompts ──────────────────────────────────────────────

TARGETED_PROMPTS = {
    "inconsistent tick intervals": (
        'Look at the axis tick marks in this chart. '
        'Read the tick values along the Y-axis (or X-axis if more prominent). '
        'Are the intervals between consecutive tick values UNEVEN? '
        'For example: 0, 10, 20, 50, 100 has uneven gaps (10, 10, 30, 50). '
        'Does this chart have "inconsistent tick intervals"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "inconsistent binning size": (
        'Is this a histogram (bars representing numeric ranges)? '
        'If yes, look at the width of each bar carefully. '
        'Are the bars DIFFERENT widths? For example, one bar covers 0-10 while another covers 10-30. '
        'Does this chart have "inconsistent binning size"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "inverted axis": (
        'Look at the Y-axis numbers in this chart. '
        'Read them from bottom to top. Do they DECREASE (e.g., bottom=100, top=0)? '
        'That would mean the axis is inverted — high values at bottom, low at top. '
        'Does this chart have an "inverted axis"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "inappropriate axis range": (
        'Look at the Y-axis range in this chart. '
        'Does it show only a very narrow slice of values, making tiny differences look huge? '
        'For example: showing 98% to 102% instead of 0% to 100%, or 4.0 to 4.5 instead of 0 to 5. '
        'Does this chart have an "inappropriate axis range"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "discretized continuous variable": (
        'Look at this chart. Is the data inherently continuous (like temperature, time, money) '
        'but displayed in discrete bins or categories that hide the true distribution? '
        'For example: showing exact ages as "20-30, 30-40" ranges when a histogram with finer bins would be more informative. '
        'Does this chart have a "discretized continuous variable"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "inappropriate item order": (
        'Look at how items are ordered in this chart. '
        'Are they arranged in a way that creates a false visual trend? '
        'For example: sorting countries by value to make it look like a declining trend, '
        'when the items have no natural sequence. '
        'Does this chart have "inappropriate item order"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
}

# ── Sample selection (bbox + content matching) ─────────────────────────────

def select_samples() -> list[dict]:
    from data_tools.misviz.loader import MisvizLoader

    b_path = Path("C:/Users/chntw/Documents/7180/PCBZ_FinChartAudit/results/claude_vision_only.json")
    b_data = json.loads(b_path.read_text(encoding="utf-8"))
    b_items = b_data["results"]

    loader = MisvizLoader()
    real_data = loader.load_real()

    local_by_bbox = {}
    for i, item in enumerate(real_data):
        bbox = item.get("bbox", [])
        if bbox:
            bk = json.dumps(bbox, sort_keys=True)
            if bk not in local_by_bbox:
                local_by_bbox[bk] = i

    matched = {}
    used = set()
    for b_item in b_items:
        bbox = b_item.get("bbox", [])
        if bbox:
            bk = json.dumps(bbox, sort_keys=True)
            if bk in local_by_bbox:
                matched[b_item["id"]] = local_by_bbox[bk]
                used.add(local_by_bbox[bk])

    content_to_local = defaultdict(list)
    for i, d in enumerate(real_data):
        if i not in used:
            key = (frozenset(d.get("misleader", [])), tuple(sorted(d.get("chart_type", []))))
            content_to_local[key].append(i)

    for b_item in b_items:
        if b_item["id"] in matched:
            continue
        key = (frozenset(b_item["gt_misleaders"]), tuple(sorted(b_item.get("chart_type", []))))
        candidates = [idx for idx in content_to_local.get(key, []) if idx not in used]
        if candidates:
            matched[b_item["id"]] = candidates[0]
            used.add(candidates[0])

    samples = []
    for b_item in b_items:
        if b_item["id"] not in matched:
            continue
        idx = matched[b_item["id"]]
        instance = loader.get_real_instance(idx)
        if Path(instance.image_path).exists():
            samples.append({
                "idx": idx,
                "instance_id": str(b_item["id"]),
                "image_path": instance.image_path,
                "ground_truth": instance.misleader,
            })

    print(f"Matched {len(samples)}/271 samples")
    return samples


# ── Helpers ─────────────────────────────────────────────────────────────────

def img_to_b64(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode in ("CMYK", "RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def extract_json(text: str) -> dict | None:
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
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def load_ocr_cache() -> dict:
    cache = {}
    if not OCR_CACHE_DIR.exists():
        return cache
    for f in sorted(OCR_CACHE_DIR.glob("*.json")):
        try:
            cache.update(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cache


def load_deplot_cache(samples: list[dict]) -> dict:
    from finchartaudit.tools.deplot import DePlotTool
    deplot = DePlotTool(device="cpu")
    cache = {}
    for s in samples:
        result = deplot._cache_load(s["image_path"])
        if result:
            cache[s["instance_id"]] = result
    return cache


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V7: sequential per-type re-ask")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from finchartaudit.config import get_config
    get_config.cache_clear()
    config = get_config()
    print(f"Model: {config.vlm_model}")

    from openai import OpenAI
    client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)

    samples = select_samples()
    ocr_cache = load_ocr_cache()
    deplot_cache = load_deplot_cache(samples)
    print(f"OCR cache: {len(ocr_cache)}, DePlot cache: {len(deplot_cache)}")

    # Resume
    results_file = OUT_DIR / "raw_results.json"
    if results_file.exists():
        existing = json.loads(results_file.read_text(encoding="utf-8"))
        done_ids = {r["instance_id"] for r in existing if "error" not in r}
        print(f"Resuming: {len(done_ids)} done")
    else:
        existing = []
        done_ids = set()

    remaining = [s for s in samples if s["instance_id"] not in done_ids]
    print(f"Remaining: {len(remaining)}/{len(samples)}")

    results = list(existing)
    lock = threading.Lock()
    completed = len(existing)
    errors = 0
    extra_calls_total = 0

    def worker(sample):
        nonlocal completed, errors, extra_calls_total
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])

        try:
            start = time.time()
            pp_log = []
            n_calls = 1

            # === Call 1: General detection ===
            resp1 = client.chat.completions.create(
                model=config.vlm_model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": CALL1_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]}],
                max_tokens=512,
            )
            raw1 = resp1.choices[0].message.content or ""
            data1 = extract_json(raw1)
            predicted = data1.get("misleader_types", []) if data1 else []
            if isinstance(predicted, str):
                predicted = [predicted]

            # === Sequential re-ask: ONLY for clean/sparse charts ===
            if len(predicted) <= 1:
                for misleader_type, prompt_text in TARGETED_PROMPTS.items():
                    if misleader_type in predicted:
                        continue

                    n_calls += 1
                    resp = client.chat.completions.create(
                        model=config.vlm_model,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ]}],
                        max_tokens=150,
                    )
                    raw = resp.choices[0].message.content or ""

                    # Parse YES/NO
                    data = extract_json(raw)
                    if data and isinstance(data.get("answer"), str) and data["answer"].upper().startswith("YES"):
                        predicted.append(misleader_type)
                        reason = data.get("reason", "")[:60]
                        pp_log.append(f"SEQ_ADD {misleader_type}: {reason}")

            elapsed = time.time() - start
            extra_calls_total += n_calls - 1

            # === Post-processing: OCR CLEAN veto ===
            ocr_data = ocr_cache.get(iid, {})
            if ocr_data and "error" not in ocr_data:
                axis_values = ocr_data.get("axis_values", [])
                right_axis_values = ocr_data.get("right_axis_values", [])
                rule_results = ocr_data.get("rule_results", [])

                new_pred = []
                for name in predicted:
                    if name == "truncated axis" and axis_values:
                        trunc_flagged = any("instead of 0" in r.lower() or "exaggerated" in r.lower()
                                            for r in rule_results if r.startswith("truncated_axis:"))
                        if not trunc_flagged and min(axis_values) <= 0:
                            pp_log.append("VETO truncated_axis")
                            continue
                    if name == "dual axis" and not right_axis_values:
                        dual_flagged = any(r.startswith("dual_axis:") for r in rule_results)
                        if not dual_flagged:
                            pp_log.append("VETO dual_axis")
                            continue
                    new_pred.append(name)
                predicted = new_pred

            # === Post-processing: DePlot axis_range ===
            deplot_data = deplot_cache.get(iid, {})
            if deplot_data and "error" not in deplot_data:
                from finchartaudit.tools.table_rules import check_inappropriate_axis_range
                rows = deplot_data.get("rows", [])
                if rows and "inappropriate axis range" not in predicted:
                    check = check_inappropriate_axis_range(rows)
                    if check.get("flagged"):
                        predicted.append("inappropriate axis range")
                        pp_log.append(f"DEPLOT_ADD axis_range")

            result = {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(elapsed, 1),
                "n_calls": n_calls,
                "pp_log": pp_log,
            }
        except Exception as e:
            result = {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": [],
                "error": str(e),
            }

        with lock:
            completed += 1
            results.append(result)
            if "error" in result:
                errors += 1
                print(f"[{completed}/{len(samples)}] id={iid} ERROR: {result['error'][:60]}")
            else:
                pp_str = f" pp={result['pp_log']}" if result.get("pp_log") else ""
                print(f"[{completed}/{len(samples)}] id={iid} "
                      f"gt={sample['ground_truth']} -> {result['predicted'][:4]} "
                      f"({result['elapsed_s']}s, {result['n_calls']}calls){pp_str}")
            if completed % 20 == 0:
                _save(results)

    print(f"\nStarting V7 sequential ({args.workers} threads)...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, s) for s in remaining]
        for future in as_completed(futures):
            try:
                future.result(timeout=300)
            except Exception as e:
                print(f"Thread error: {e}")

    _save(results)
    total_time = time.time() - t0

    # Metrics
    from data_tools.misviz.evaluator import MisvizEvaluator
    evaluator = MisvizEvaluator()
    for r in results:
        if "error" not in r:
            evaluator.add_prediction(
                instance_id=r["instance_id"],
                ground_truth=r["ground_truth"],
                predicted=r["predicted"],
                condition="v7_sequential",
                model="claude_haiku",
            )

    evaluator.print_summary()
    metrics = evaluator.compute_metrics()
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    avg_calls = sum(r.get("n_calls", 1) for r in results) / len(results)
    print(f"\nDone: {completed} charts, {errors} errors")
    print(f"Avg calls/chart: {avg_calls:.1f}, Extra calls: {extra_calls_total}")
    print(f"Time: {total_time:.0f}s ({total_time/max(completed,1):.1f}s/chart)")


def _save(results):
    (OUT_DIR / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
