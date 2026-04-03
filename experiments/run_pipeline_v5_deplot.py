"""V5 pipeline: B's prompt + DePlot table rules + CLEAN veto.

Phase 1: DePlot extracts data tables from all charts (GPU, cached)
Phase 2: VLM-only detection (B's prompt, no OCR/rule injection)
Phase 3: Post-processing — OCR CLEAN veto + DePlot table rule checks

Usage:
    python run_pipeline_v5_deplot.py                     # Full run
    python run_pipeline_v5_deplot.py --phase 1           # DePlot only
    python run_pipeline_v5_deplot.py --phase 2           # VLM + post-processing
    python run_pipeline_v5_deplot.py --workers 4         # Fewer VLM threads
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

OUT_DIR = Path("data/eval_results/v5_deplot")
OCR_CACHE_DIR = Path("data/eval_results/pipeline_full/ocr_cache")

# ── B's prompt + disambiguation few-shots (same as V4) ─────────────────────

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
{"misleading": true, "misleader_types": ["3d"], "explanation": "3D perspective distorts visual comparison of bar sizes. This is a 3D rendering issue, not misrepresentation — the labeled values may be correct but the visual encoding is distorted by perspective."}

EXAMPLE 5
Chart: A line chart showing stock price from $45 to $52 over 6 months. The y-axis starts at $44.
Output:
{"misleading": false, "misleader_types": [], "explanation": "Line charts commonly use non-zero y-axis baselines to show trends clearly. Starting at $44 for a $45-$52 range is standard practice, not truncation."}

EXAMPLE 6
Chart: A bar chart showing satisfaction scores from 4.1 to 4.5 on a scale of 1-5. The y-axis starts at 4.0.
Output:
{"misleading": true, "misleader_types": ["truncated axis"], "explanation": "The bar chart y-axis starts at 4.0 instead of 0, making small differences between 4.1 and 4.5 appear much larger than they are."}
""".strip()

OUTPUT_FORMAT = """{
  "misleading": <true|false>,
  "misleader_types": [<zero or more types from the taxonomy>],
  "explanation": "<one to three sentences>"
}"""

USER_PROMPT = f"""You are an expert in data visualization. Detect misleading elements in the chart image.

## Misleader Taxonomy
{TAXONOMY_BLOCK}

## Examples
{FEW_SHOT_EXAMPLES}

## Output
Respond with valid JSON only:
{OUTPUT_FORMAT}"""


# ── Sample selection ────────────────────────────────────────────────────────

def select_samples() -> list[dict]:
    from data_tools.misviz.loader import MisvizLoader

    b_path = Path("C:/Users/chntw/Documents/7180/PCBZ_FinChartAudit/results/claude_vision_only.json")
    b_data = json.loads(b_path.read_text(encoding="utf-8"))
    b_items = b_data["results"]

    loader = MisvizLoader()
    real_data = loader.load_real()

    # Step 1: bbox matching
    local_by_bbox = {}
    for i, item in enumerate(real_data):
        bbox = item.get("bbox", [])
        if bbox:
            bbox_key = json.dumps(bbox, sort_keys=True)
            if bbox_key not in local_by_bbox:
                local_by_bbox[bbox_key] = i

    matched = {}
    used = set()
    for b_item in b_items:
        bbox = b_item.get("bbox", [])
        if bbox:
            bbox_key = json.dumps(bbox, sort_keys=True)
            if bbox_key in local_by_bbox:
                idx = local_by_bbox[bbox_key]
                matched[b_item["id"]] = idx
                used.add(idx)

    # Step 2: content matching for remaining
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
            idx = candidates[0]
            matched[b_item["id"]] = idx
            used.add(idx)

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

    print(f"Matched {len(samples)}/271 samples (bbox: {sum(1 for b in b_items if b.get('bbox') and b['id'] in matched)})")
    return samples


# ── Phase 1: DePlot batch extraction ───────────────────────────────────────

def run_phase1(samples: list[dict]):
    """Extract data tables from all chart images using DePlot (GPU)."""
    from finchartaudit.tools.deplot import DePlotTool

    deplot = DePlotTool(device="auto")
    print(f"DePlot cache: {deplot._cache_dir}")

    total = len(samples)
    for i, s in enumerate(samples):
        result = deplot.extract_table(s["image_path"])
        has_data = len(result.get("values", [])) > 0
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{total}] data={'yes' if has_data else 'no'} vals={len(result.get('values', []))}")

    print(f"\nDePlot done: {deplot.cache_stats}")


# ── Phase 2: VLM + post-processing ─────────────────────────────────────────

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
    """Load DePlot results for all samples."""
    from finchartaudit.tools.deplot import DePlotTool
    deplot = DePlotTool(device="cpu")  # Only loading cache, no inference
    cache = {}
    for s in samples:
        result = deplot._cache_load(s["image_path"])
        if result:
            cache[s["instance_id"]] = result
    return cache


def apply_postprocessing(predicted: list[str], ocr_data: dict, deplot_data: dict) -> tuple[list[str], list[str]]:
    """Combined post-processing: OCR CLEAN veto + DePlot table rule additions.

    1. OCR CLEAN veto: remove FP for truncated_axis and dual_axis
    2. DePlot rules: add detections VLM missed (tick intervals, binning, inverted, axis range)
    """
    from finchartaudit.tools.table_rules import analyze_deplot_table

    veto_log = []
    result = list(predicted)

    # --- OCR CLEAN veto (same as V4) ---
    if ocr_data and "error" not in ocr_data:
        axis_values = ocr_data.get("axis_values", [])
        right_axis_values = ocr_data.get("right_axis_values", [])
        rule_results = ocr_data.get("rule_results", [])

        new_result = []
        for name in result:
            if name == "truncated axis" and axis_values:
                trunc_flagged = any("instead of 0" in r.lower() or "exaggerated" in r.lower()
                                    for r in rule_results if r.startswith("truncated_axis:"))
                if not trunc_flagged and min(axis_values) <= 0:
                    veto_log.append(f"VETO truncated_axis: axis includes 0 (min={min(axis_values)})")
                    continue
            if name == "dual axis" and not right_axis_values:
                dual_flagged = any(r.startswith("dual_axis:") for r in rule_results)
                if not dual_flagged:
                    veto_log.append("VETO dual_axis: no right Y-axis in OCR")
                    continue
            new_result.append(name)
        result = new_result

    # --- DePlot table rule additions ---
    if deplot_data and "error" not in deplot_data:
        table_checks = analyze_deplot_table(deplot_data)

        # Add inconsistent tick intervals if VLM missed and rule flagged
        if "inconsistent tick intervals" not in result:
            tick_check = table_checks.get("tick_intervals", {})
            if tick_check.get("flagged") and tick_check.get("max_deviation", 0) > 0.3:
                result.append("inconsistent tick intervals")
                veto_log.append(f"ADD tick_intervals: deviation={tick_check['max_deviation']:.1%}")

        # Add inconsistent binning if VLM missed and rule flagged
        if "inconsistent binning size" not in result:
            bin_check = table_checks.get("binning", {})
            if bin_check.get("flagged") and bin_check.get("max_deviation", 0) > 0.4:
                result.append("inconsistent binning size")
                veto_log.append(f"ADD binning: deviation={bin_check['max_deviation']:.1%}")

        # Add inverted axis if VLM missed and rule flagged
        if "inverted axis" not in result:
            inv_check = table_checks.get("inverted_axis", {})
            if inv_check.get("flagged"):
                result.append("inverted axis")
                veto_log.append(f"ADD inverted_axis: {inv_check.get('reason', '')}")

        # Add inappropriate axis range if VLM missed and rule flagged
        if "inappropriate axis range" not in result:
            range_check = table_checks.get("axis_range", {})
            if range_check.get("flagged"):
                result.append("inappropriate axis range")
                veto_log.append(f"ADD axis_range: ratio={range_check.get('range_ratio', '?')}")

    return result, veto_log


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


def img_to_b64(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode in ("CMYK", "RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def run_phase2(samples: list[dict], workers: int = 8):
    """VLM detection + combined post-processing."""
    from finchartaudit.config import get_config
    from data_tools.misviz.evaluator import MisvizEvaluator

    get_config.cache_clear()
    config = get_config()
    print(f"Model: {config.vlm_model}")

    ocr_cache = load_ocr_cache()
    deplot_cache = load_deplot_cache(samples)
    print(f"OCR cache: {len(ocr_cache)}, DePlot cache: {len(deplot_cache)}")

    from openai import OpenAI
    client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)

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
    errors = sum(1 for r in existing if "error" in r)

    def worker(sample):
        nonlocal completed, errors
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=config.vlm_model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]}],
                max_tokens=512,
            )
            elapsed = time.time() - start
            raw_text = response.choices[0].message.content or ""

            data = extract_json(raw_text)
            predicted = data.get("misleader_types", []) if data else []
            if isinstance(predicted, str):
                predicted = [predicted]

            # Post-processing
            ocr_data = ocr_cache.get(iid, {})
            deplot_data = deplot_cache.get(iid, {})
            predicted, pp_log = apply_postprocessing(predicted, ocr_data, deplot_data)

            result = {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(elapsed, 1),
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
                      f"gt={sample['ground_truth']} -> {result['predicted']} "
                      f"({result['elapsed_s']}s){pp_str}")
            if completed % 20 == 0:
                _save(results)

    print(f"\nStarting V5 ({workers} threads)...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, s) for s in remaining]
        for future in as_completed(futures):
            try:
                future.result(timeout=120)
            except Exception as e:
                print(f"Thread error: {e}")

    _save(results)

    # Metrics
    evaluator = MisvizEvaluator()
    for r in results:
        if "error" not in r:
            evaluator.add_prediction(
                instance_id=r["instance_id"],
                ground_truth=r["ground_truth"],
                predicted=r["predicted"],
                condition="v5_deplot",
                model="claude_haiku",
            )

    evaluator.print_summary()
    metrics = evaluator.compute_metrics()
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nDone: {completed} charts, {errors} errors, {time.time() - t0:.0f}s")


def _save(results):
    (OUT_DIR / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="V5: B's prompt + DePlot rules + CLEAN veto")
    parser.add_argument("--phase", type=int, default=0, help="1=DePlot only, 2=VLM+PP, 0=both")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = select_samples()

    if args.phase in (0, 1):
        print(f"\n{'='*60}")
        print("PHASE 1: DePlot extraction (GPU)")
        print(f"{'='*60}")
        t1 = time.time()
        run_phase1(samples)
        print(f"Phase 1: {time.time() - t1:.0f}s")

    if args.phase in (0, 2):
        print(f"\n{'='*60}")
        print("PHASE 2: VLM detection + post-processing")
        print(f"{'='*60}")
        run_phase2(samples, workers=args.workers)


if __name__ == "__main__":
    main()
