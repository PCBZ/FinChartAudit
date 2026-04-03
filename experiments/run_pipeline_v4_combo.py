"""V4 combo: B's original prompt + disambiguation few-shots + CLEAN veto.

Strategy:
1. Use B's prompt structure (no system prompt, all in user message, with few-shots)
2. Add 2-3 disambiguation examples targeting top FP sources
3. Post-processing: CLEAN veto from cached OCR/rule data

Usage:
    python run_pipeline_v4_combo.py              # Full run
    python run_pipeline_v4_combo.py --workers 4  # Fewer threads
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

OUT_DIR = Path("data/eval_results/v4_combo")
OCR_CACHE_DIR = Path("data/eval_results/pipeline_full/ocr_cache")

# ── B's prompt structure + disambiguation few-shots ─────────────────────────

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

# B's original 3 examples + our 3 disambiguation examples
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


# ── Sample selection (same as B) ────────────────────────────────────────────

def select_samples() -> list[dict]:
    from data_tools.misviz.loader import MisvizLoader

    b_results_path = Path("C:/Users/chntw/Documents/7180/PCBZ_FinChartAudit/results/claude_vision_only.json")
    b_data = json.loads(b_results_path.read_text(encoding="utf-8"))
    b_items = b_data["results"]

    loader = MisvizLoader()
    real_data = loader.load_real()

    content_to_local = defaultdict(list)
    for i, d in enumerate(real_data):
        key = (frozenset(d.get("misleader", [])), tuple(sorted(d.get("chart_type", []))))
        content_to_local[key].append(i)

    samples = []
    used = set()
    for b_item in b_items:
        key = (frozenset(b_item["gt_misleaders"]), tuple(sorted(b_item.get("chart_type", []))))
        candidates = [idx for idx in content_to_local.get(key, []) if idx not in used]
        if candidates:
            idx = candidates[0]
            used.add(idx)
            instance = loader.get_real_instance(idx)
            if Path(instance.image_path).exists():
                samples.append({
                    "idx": idx,
                    "instance_id": str(b_item["id"]),
                    "image_path": instance.image_path,
                    "ground_truth": instance.misleader,
                    "b_id": b_item["id"],
                })
    print(f"Matched {len(samples)}/271 of B's samples")
    return samples


# ── OCR cache + CLEAN veto ──────────────────────────────────────────────────

def load_ocr_cache() -> dict:
    cache = {}
    if not OCR_CACHE_DIR.exists():
        return cache
    for f in sorted(OCR_CACHE_DIR.glob("*.json")):
        try:
            batch = json.loads(f.read_text(encoding="utf-8"))
            cache.update(batch)
        except Exception:
            pass
    return cache


def apply_clean_veto(predicted: list[str], ocr_data: dict) -> tuple[list[str], list[str]]:
    """Veto VLM predictions using only [CLEAN] rule verdicts."""
    if not ocr_data or "error" in ocr_data:
        return predicted, []

    axis_values = ocr_data.get("axis_values", [])
    right_axis_values = ocr_data.get("right_axis_values", [])
    rule_results = ocr_data.get("rule_results", [])

    veto_log = []
    vetoed = []

    for name in predicted:
        # truncated axis: veto if rule confirmed axis includes 0
        if name == "truncated axis" and axis_values:
            trunc_flagged = any("instead of 0" in r.lower() or "exaggerated" in r.lower()
                                for r in rule_results if r.startswith("truncated_axis:"))
            if not trunc_flagged and min(axis_values) <= 0:
                veto_log.append(f"VETO truncated_axis: axis includes 0 (min={min(axis_values)})")
                continue

        # dual axis: veto if OCR found no right Y-axis values
        if name == "dual axis":
            if not right_axis_values:
                dual_flagged = any(r.startswith("dual_axis:") for r in rule_results)
                if not dual_flagged:
                    veto_log.append("VETO dual_axis: no right Y-axis detected by OCR")
                    continue

        vetoed.append(name)

    return vetoed, veto_log


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


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V4 combo: B's prompt + few-shots + CLEAN veto")
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
    print(f"OCR cache: {len(ocr_cache)} entries")

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
    total_vetoes = 0

    def worker(sample):
        nonlocal completed, errors, total_vetoes
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])

        try:
            start = time.time()
            # B's structure: no system prompt, all in user message
            response = client.chat.completions.create(
                model=config.vlm_model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ]},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            elapsed = time.time() - start
            raw_text = response.choices[0].message.content or ""

            data = extract_json(raw_text)
            if data:
                predicted = data.get("misleader_types", [])
                if isinstance(predicted, str):
                    predicted = [predicted]
            else:
                predicted = []

            # CLEAN veto
            ocr_data = ocr_cache.get(iid, {})
            predicted, veto_log = apply_clean_veto(predicted, ocr_data)

            result = {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(elapsed, 1),
                "veto_log": veto_log,
                "explanation": data.get("explanation", "") if data else "",
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
            if result.get("veto_log"):
                total_vetoes += len(result["veto_log"])
            if "error" in result:
                errors += 1
                print(f"[{completed}/{len(samples)}] id={iid} ERROR: {result['error'][:60]}")
            else:
                veto_str = f" vetoed={result['veto_log']}" if result.get("veto_log") else ""
                print(f"[{completed}/{len(samples)}] id={iid} "
                      f"gt={sample['ground_truth']} -> {result['predicted']} "
                      f"({result['elapsed_s']}s){veto_str}")
            if completed % 20 == 0:
                _save(results)

    print(f"\nStarting V4 combo ({args.workers} threads)...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, s) for s in remaining]
        for future in as_completed(futures):
            try:
                future.result(timeout=120)
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
                condition="v4_combo",
                model="claude_haiku",
            )

    evaluator.print_summary()
    metrics = evaluator.compute_metrics()
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nDone: {completed} charts, {errors} errors, {total_vetoes} vetoes applied")
    print(f"Time: {total_time:.0f}s ({total_time/max(completed,1):.1f}s/chart)")


def _save(results):
    (OUT_DIR / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
