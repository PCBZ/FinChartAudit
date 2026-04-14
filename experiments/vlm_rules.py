"""VLM + Rules Pipeline experiment on Misviz real dataset (aligned with B's 271 samples).

Two-call approach, no OCR needed:
  VLM Call 1: Extract structured data (axis values, chart type) from image
  Rule Engine: Deterministic checks on extracted values
  VLM Call 2: Final judgment with rule evidence injected

Usage:
    python run_vlm_rules.py                  # Full run (8 threads)
    python run_vlm_rules.py --workers 4      # Fewer threads
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from finchartaudit.config import get_config
from finchartaudit.tools.rule_check import RuleEngine
from finchartaudit.prompts.t2_visual import MISLEADER_DEFINITIONS, COMPLETENESS_CHECKS, SEC_RULE_MAPPING
from finchartaudit.agents.t2_pipeline import PIPELINE_SYSTEM_PROMPT, PIPELINE_PROMPT

OUT_DIR = Path("data/eval_results/vlm_rules")

# ── Prompts ──────────────────────────────────────────────────────────────────

EXTRACT_SYSTEM = "You are a precise data extraction assistant. Extract exactly what is asked, nothing more."

EXTRACT_PROMPT = """Look at this chart image carefully and extract the following structured data.

Return ONLY valid JSON with these fields:
{
  "chart_type": "bar|line|pie|area|scatter|histogram|other",
  "y_axis_values": [list of numeric values on the Y-axis, read from top to bottom, e.g. [100, 80, 60, 40, 20, 0]],
  "x_axis_values": [list of numeric values on the X-axis if numeric, else []],
  "has_right_y_axis": true/false,
  "right_y_axis_values": [list of numeric values on right Y-axis if present, else []],
  "has_3d_effect": true/false,
  "num_data_series": 1
}

Rules:
- For y_axis_values: read ALL tick mark numbers on the left Y-axis, top to bottom
- For x_axis_values: only include if they are NUMBERS (not dates, not categories)
- If you cannot read a value clearly, skip it
- Return valid JSON only, no explanation"""


# ── Sample selection (same as run_pipeline_full.py) ──────────────────────────

def select_samples_aligned_with_b() -> list[dict]:
    """Use B's exact 271 sample IDs from his results file."""
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


# ── Core logic ───────────────────────────────────────────────────────────────

def img_to_b64(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode in ("CMYK", "RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def vlm_call(client, model: str, image_b64: str, system: str, prompt: str) -> str:
    """Single VLM API call."""
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


def run_rule_checks(engine: RuleEngine, extracted: dict) -> list[str]:
    """Run all applicable rule checks on VLM-extracted data."""
    results = []
    y_values = extracted.get("y_axis_values", [])
    x_values = extracted.get("x_axis_values", [])
    right_values = extracted.get("right_y_axis_values", [])
    chart_type = extracted.get("chart_type", "bar")

    # Ensure numeric
    y_values = [float(v) for v in y_values if isinstance(v, (int, float))]
    x_values = [float(v) for v in x_values if isinstance(v, (int, float))]
    right_values = [float(v) for v in right_values if isinstance(v, (int, float))]

    if y_values:
        # Truncated axis
        try:
            r = engine.run_check("truncated_axis", {"axis_values": y_values, "chart_type": chart_type})
            results.append(f"truncated_axis: {r['explanation']}")
        except Exception:
            pass

        # Broken scale / inconsistent tick intervals
        try:
            r = engine.run_check("broken_scale", {"axis_values": y_values})
            results.append(f"broken_scale: {r['explanation']}")
        except Exception:
            pass

        # Inverted axis
        try:
            r = engine.run_check("inverted_axis", {"axis_values": y_values})
            if r["is_inverted"]:
                results.append(f"inverted_axis: {r['explanation']}")
        except Exception:
            pass

        # Inappropriate axis range
        try:
            r = engine.run_check("inappropriate_axis_range", {"axis_values": y_values})
            if r["is_inappropriate"]:
                results.append(f"inappropriate_axis_range: {r['explanation']}")
        except Exception:
            pass

    # Dual axis
    if y_values and right_values:
        try:
            r = engine.run_check("dual_axis", {
                "left_axis_values": y_values, "right_axis_values": right_values})
            if r["has_dual_axis"]:
                results.append(f"dual_axis: {r['explanation']}")
        except Exception:
            pass

    # Inconsistent binning (from X-axis)
    if len(x_values) >= 3:
        try:
            r = engine.run_check("inconsistent_binning", {"bin_edges": x_values})
            if r["is_inconsistent"]:
                results.append(f"inconsistent_binning: {r['explanation']}")
        except Exception:
            pass

    return results


def process_one(client, model: str, engine: RuleEngine, sample: dict) -> dict:
    """Process one chart: VLM extract → Rules → VLM judge."""
    iid = sample["instance_id"]
    image_b64 = img_to_b64(sample["image_path"])
    start = time.time()

    # Call 1: Extract structured data
    raw_extract = vlm_call(client, model, image_b64, EXTRACT_SYSTEM, EXTRACT_PROMPT)
    extracted = extract_json(raw_extract) or {}
    t_extract = time.time() - start

    # Rule checks
    rule_results = run_rule_checks(engine, extracted)

    # Build evidence string for Call 2
    axis_str = f"Y-axis values (top to bottom): {extracted.get('y_axis_values', [])}"
    if extracted.get("right_y_axis_values"):
        axis_str += f"\nRight Y-axis values: {extracted['right_y_axis_values']}"

    # Call 2: Final judgment with rule evidence
    misleader_list = "\n".join(f"- {k}: {v}" for k, v in MISLEADER_DEFINITIONS.items())
    completeness_list = "\n".join(f"- {k}: {v}" for k, v in COMPLETENESS_CHECKS.items())

    judge_prompt = PIPELINE_PROMPT.format(
        chart_id=f"eval_{iid}",
        page=1,
        ocr_text=f"Chart type: {extracted.get('chart_type', 'unknown')}\n{axis_str}",
        ocr_axis=str(extracted.get("y_axis_values", [])),
        rule_results="\n".join(rule_results) if rule_results else "No rule violations detected.",
        misleader_list=misleader_list,
        completeness_list=completeness_list,
    )

    raw_judge = vlm_call(client, model, image_b64, PIPELINE_SYSTEM_PROMPT, judge_prompt)
    elapsed = time.time() - start

    # Parse predictions
    judge_data = extract_json(raw_judge) or {}
    predicted = []
    for name, assessment in judge_data.get("misleaders", {}).items():
        if isinstance(assessment, dict) and assessment.get("present"):
            confidence = float(assessment.get("confidence", 0))
            if confidence >= 0.3:
                predicted.append(name)

    return {
        "instance_id": iid,
        "ground_truth": sample["ground_truth"],
        "predicted": list(set(predicted)),
        "elapsed_s": round(elapsed, 1),
        "t_extract_s": round(t_extract, 1),
        "extracted_chart_type": extracted.get("chart_type", ""),
        "extracted_y_values": extracted.get("y_axis_values", []),
        "rule_results": rule_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLM + Rules Pipeline experiment")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    get_config.cache_clear()
    config = get_config()
    print(f"Model: {config.vlm_model}")

    from openai import OpenAI
    client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)
    engine = RuleEngine()

    samples = select_samples_aligned_with_b()

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

    def worker(sample):
        nonlocal completed, errors
        try:
            result = process_one(client, config.vlm_model, engine, sample)
        except Exception as e:
            result = {
                "instance_id": sample["instance_id"],
                "ground_truth": sample["ground_truth"],
                "predicted": [],
                "error": str(e),
            }

        with lock:
            completed += 1
            results.append(result)
            if "error" in result:
                errors += 1
                print(f"[{completed}/{len(samples)}] id={result['instance_id']} ERROR: {result['error'][:60]}")
            else:
                print(f"[{completed}/{len(samples)}] id={result['instance_id']} "
                      f"gt={result['ground_truth']} -> {result['predicted']} "
                      f"({result['elapsed_s']}s, rules={len(result.get('rule_results', []))})")
            if completed % 20 == 0:
                _save(results)

    print(f"\nStarting {args.workers}-thread VLM+Rules pipeline...")
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
                condition="vlm_rules",
                model="claude_haiku",
            )

    evaluator.print_summary()
    metrics = evaluator.compute_metrics()
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nDone: {completed} charts, {errors} errors, {total_time:.0f}s total")
    print(f"Avg: {total_time / max(completed, 1):.1f}s/chart (wall clock)")
    print(f"Results: {OUT_DIR}")


def _save(results):
    (OUT_DIR / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
