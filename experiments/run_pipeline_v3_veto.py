"""Route 3 experiment: VLM-only detection + Rule [CLEAN] post-processing veto.

VLM sees ONLY the image (no OCR data in prompt).
After VLM returns predictions, apply rule veto using cached OCR/rule data:
  - truncated_axis [CLEAN] → veto VLM's "truncated axis" prediction
  - dual_axis [CLEAN] → veto VLM's "dual axis" prediction

Usage:
    python run_pipeline_v3_veto.py              # Full run
    python run_pipeline_v3_veto.py --workers 4  # Fewer threads
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

OUT_DIR = Path("data/eval_results/vlm_only_veto")
OCR_CACHE_DIR = Path("data/eval_results/pipeline_full/ocr_cache")

# ── Vision-only prompt (no OCR, no rules) ───────────────────────────────────

SYSTEM_PROMPT = """You are a financial chart auditor detecting misleading visual encodings.

Analyze the chart image carefully. For each potential issue, you must have clear visual evidence.

Key calibration:
- Line/scatter charts are ALLOWED to have non-zero Y-axis origins.
- Only flag "misrepresentation" if specific bar heights/pie angles clearly don't match their data labels.
- Only flag "truncated axis" for bar/area charts where Y-axis clearly doesn't start at 0.
- Don't flag issues you're unsure about — precision matters more than recall."""

USER_PROMPT = """Analyze this chart image for misleading visual elements.

Check for these issues (only flag what you can clearly see):
- truncated axis: Y-axis doesn't start at 0 in a bar/area chart
- misrepresentation: bar heights/pie angles don't match labeled values
- 3d: 3D perspective distorts value perception
- dual axis: two Y-axes with different scales
- inverted axis: axis values run in reverse order
- inappropriate axis range: Y-axis range exaggerates tiny differences
- inconsistent tick intervals: tick marks not evenly spaced
- inconsistent binning size: histogram bins with unequal widths
- discretized continuous variable: continuous data forced into categories
- inappropriate use of pie chart: pie chart for non-part-of-whole data
- inappropriate use of line chart: line chart for categorical data
- inappropriate item order: ordering creates false trend impression

Respond with ONLY valid JSON:
{
  "misleading": true/false,
  "misleader_types": ["list of detected types, empty if clean"],
  "explanation": "one to three sentences with specific evidence"
}"""


# ── Sample selection ────────────────────────────────────────────────────────

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


# ── OCR cache loading ───────────────────────────────────────────────────────

def load_ocr_cache() -> dict:
    cache = {}
    if not OCR_CACHE_DIR.exists():
        print("WARNING: No OCR cache found. Run run_pipeline_full.py --phase 1 first.")
        return cache
    for f in sorted(OCR_CACHE_DIR.glob("*.json")):
        try:
            batch = json.loads(f.read_text(encoding="utf-8"))
            cache.update(batch)
        except Exception:
            pass
    return cache


# ── Rule [CLEAN] veto ───────────────────────────────────────────────────────

def apply_clean_veto(predicted: list[str], ocr_data: dict) -> tuple[list[str], list[str]]:
    """Veto VLM predictions using only [CLEAN] rule verdicts.

    Only vetoes when rule is CONFIDENT the issue does NOT exist.
    Never adds predictions — only removes false positives.

    Returns (vetoed_predictions, veto_log)
    """
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
    parser = argparse.ArgumentParser(description="VLM-only + Rule CLEAN veto")
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
    print(f"OCR cache: {len(ocr_cache)} entries (for post-processing veto)")

    # Resume support
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
            response = client.chat.completions.create(
                model=config.vlm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ]},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            elapsed = time.time() - start
            raw_text = response.choices[0].message.content or ""

            # Parse VLM response
            data = extract_json(raw_text)
            if data:
                predicted = data.get("misleader_types", [])
                if isinstance(predicted, str):
                    predicted = [predicted]
            else:
                predicted = []

            # Apply rule [CLEAN] veto
            ocr_data = ocr_cache.get(iid, {})
            predicted, veto_log = apply_clean_veto(predicted, ocr_data)

            result = {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(elapsed, 1),
                "veto_log": veto_log,
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

    print(f"\nStarting {args.workers}-thread VLM-only + veto...")
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

    # Compute metrics
    from data_tools.misviz.evaluator import MisvizEvaluator
    evaluator = MisvizEvaluator()
    for r in results:
        if "error" not in r:
            evaluator.add_prediction(
                instance_id=r["instance_id"],
                ground_truth=r["ground_truth"],
                predicted=r["predicted"],
                condition="vlm_only_veto",
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
