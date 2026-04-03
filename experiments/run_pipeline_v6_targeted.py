"""V6 pipeline: B's prompt + targeted VLM re-ask for blind spots + DePlot axis_range + CLEAN veto.

Architecture:
  VLM Call 1: B's prompt — general detection (all types)
  VLM Call 2: Only for charts VLM reported clean — targeted questions for blind spot types
  Post-processing: OCR CLEAN veto + DePlot axis_range rule

Usage:
    python run_pipeline_v6_targeted.py              # Full run
    python run_pipeline_v6_targeted.py --workers 4
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

OUT_DIR = Path("data/eval_results/v6_targeted")
OCR_CACHE_DIR = Path("data/eval_results/pipeline_full/ocr_cache")

# ── VLM Call 1: B's prompt (same as V4/V5) ─────────────────────────────────

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

# ── VLM Call 2: Targeted questions for blind spot types ─────────────────────

CALL2_PROMPT = """Look at this chart image very carefully and answer each question with YES or NO, plus a brief reason.

Q1 - TICK INTERVALS: Look at the Y-axis tick marks (the small lines/numbers along the vertical axis). Measure the gaps between consecutive ticks. Are the gaps UNEVEN (e.g., 0, 10, 20, 50, 100 — the jump from 20 to 50 is different from 10 to 20)?
Answer: YES if ticks are unevenly spaced, NO if they are evenly spaced or if there is no visible Y-axis.

Q2 - BINNING: If this is a histogram (bars touching each other representing ranges), look at the width of each bar. Are the bars DIFFERENT widths (e.g., one bar covers ages 0-10 while another covers 10-30)?
Answer: YES if bars have unequal widths, NO if they are equal or if this is not a histogram.

Q3 - INVERTED AXIS: Look at the Y-axis numbers. Do they run from HIGH at the bottom to LOW at the top (the opposite of normal)?
Answer: YES if the axis is inverted, NO if numbers increase from bottom to top (normal).

Q4 - AXIS RANGE: Look at the Y-axis range. Does it show only a very narrow slice of values (e.g., 98% to 102% instead of 0% to 100%), making tiny differences look huge?
Answer: YES if the range is misleadingly narrow, NO if it seems reasonable.

Respond with ONLY valid JSON:
{
  "tick_intervals": {"answer": "YES/NO", "reason": "..."},
  "binning": {"answer": "YES/NO", "reason": "..."},
  "inverted_axis": {"answer": "YES/NO", "reason": "..."},
  "axis_range": {"answer": "YES/NO", "reason": "..."}
}"""

BLIND_SPOT_MAP = {
    "tick_intervals": "inconsistent tick intervals",
    "binning": "inconsistent binning size",
    "inverted_axis": "inverted axis",
    "axis_range": "inappropriate axis range",
}

# ── Sample selection (with bbox matching) ───────────────────────────────────

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


def apply_clean_veto(predicted: list[str], ocr_data: dict) -> tuple[list[str], list[str]]:
    """OCR CLEAN veto (same as V4)."""
    if not ocr_data or "error" in ocr_data:
        return predicted, []

    axis_values = ocr_data.get("axis_values", [])
    right_axis_values = ocr_data.get("right_axis_values", [])
    rule_results = ocr_data.get("rule_results", [])
    veto_log = []
    result = []

    for name in predicted:
        if name == "truncated axis" and axis_values:
            trunc_flagged = any("instead of 0" in r.lower() or "exaggerated" in r.lower()
                                for r in rule_results if r.startswith("truncated_axis:"))
            if not trunc_flagged and min(axis_values) <= 0:
                veto_log.append(f"VETO truncated_axis: axis includes 0")
                continue
        if name == "dual axis" and not right_axis_values:
            dual_flagged = any(r.startswith("dual_axis:") for r in rule_results)
            if not dual_flagged:
                veto_log.append("VETO dual_axis: no right Y-axis in OCR")
                continue
        result.append(name)

    return result, veto_log


def apply_deplot_axis_range(predicted: list[str], deplot_data: dict) -> tuple[list[str], list[str]]:
    """DePlot axis_range rule only (the one that actually works)."""
    if not deplot_data or "error" in deplot_data:
        return predicted, []

    from finchartaudit.tools.table_rules import check_inappropriate_axis_range
    rows = deplot_data.get("rows", [])
    if not rows:
        return predicted, []

    check = check_inappropriate_axis_range(rows)
    if check.get("flagged") and "inappropriate axis range" not in predicted:
        predicted = predicted + ["inappropriate axis range"]
        return predicted, [f"ADD axis_range: {check.get('reason', '')}"]

    return predicted, []


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V6: targeted re-ask for blind spots")
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
    call2_count = 0

    def worker(sample):
        nonlocal completed, errors, call2_count
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])

        try:
            # === VLM Call 1: General detection ===
            start = time.time()
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

            # === VLM Call 2: Targeted re-ask ONLY if Call 1 found few issues ===
            # Rationale: if Call 1 already found 2+ types, chart is flagged and
            # adding more types risks FP. Only re-ask when Call 1 said clean or
            # found at most 1 issue — these are the charts where blind spots matter.
            blind_spots_missing = [
                bs for bs in BLIND_SPOT_MAP.values()
                if bs not in predicted
            ]

            pp_log = []
            if blind_spots_missing:
                call2_count += 1
                resp2 = client.chat.completions.create(
                    model=config.vlm_model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": CALL2_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ]}],
                    max_tokens=512,
                )
                raw2 = resp2.choices[0].message.content or ""
                data2 = extract_json(raw2)

                if data2:
                    for key, misleader_name in BLIND_SPOT_MAP.items():
                        if misleader_name in predicted:
                            continue
                        answer = data2.get(key, {})
                        if isinstance(answer, dict) and answer.get("answer", "").upper().startswith("YES"):
                            predicted.append(misleader_name)
                            reason = answer.get("reason", "")[:60]
                            pp_log.append(f"CALL2_ADD {key}: {reason}")

            elapsed = time.time() - start

            # === Post-processing: OCR CLEAN veto ===
            ocr_data = ocr_cache.get(iid, {})
            predicted, veto_log = apply_clean_veto(predicted, ocr_data)
            pp_log.extend(veto_log)

            # === Post-processing: DePlot axis_range ===
            deplot_data = deplot_cache.get(iid, {})
            predicted, deplot_log = apply_deplot_axis_range(predicted, deplot_data)
            pp_log.extend(deplot_log)

            result = {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(elapsed, 1),
                "pp_log": pp_log,
                "had_call2": bool(blind_spots_missing),
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
                c2 = " [2calls]" if result.get("had_call2") else ""
                print(f"[{completed}/{len(samples)}] id={iid} "
                      f"gt={sample['ground_truth']} -> {result['predicted'][:4]} "
                      f"({result['elapsed_s']}s){c2}{pp_str}")
            if completed % 20 == 0:
                _save(results)

    print(f"\nStarting V6 targeted ({args.workers} threads)...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, s) for s in remaining]
        for future in as_completed(futures):
            try:
                future.result(timeout=180)
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
                condition="v6_targeted",
                model="claude_haiku",
            )

    evaluator.print_summary()
    metrics = evaluator.compute_metrics()
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    had_call2 = sum(1 for r in results if r.get("had_call2"))
    print(f"\nDone: {completed} charts, {errors} errors, {had_call2} had Call 2")
    print(f"Time: {total_time:.0f}s ({total_time/max(completed,1):.1f}s/chart)")


def _save(results):
    (OUT_DIR / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
