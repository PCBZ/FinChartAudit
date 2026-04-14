"""LLM-OCR + Rules Pipeline experiment on Misviz real dataset (aligned with B's 271 samples).

Uses VLM as a pure OCR tool (read text only, no interpretation), then rule engine,
then a second VLM call for final judgment.

  VLM Call 1: Pure OCR — read all visible text from image regions  ~5s
  Rule Engine: Deterministic checks on extracted numbers            <1s
  VLM Call 2: Final judgment with rule evidence injected            ~5s

Usage:
    python run_llm_ocr_rules.py                  # Full run (8 threads)
    python run_llm_ocr_rules.py --workers 4      # Fewer threads
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

from finchartaudit.config import get_config
from finchartaudit.tools.rule_check import RuleEngine
from finchartaudit.prompts.t2_visual import MISLEADER_DEFINITIONS, COMPLETENESS_CHECKS
from finchartaudit.agents.t2_pipeline import PIPELINE_SYSTEM_PROMPT, PIPELINE_PROMPT

OUT_DIR = Path("data/eval_results/llm_ocr_rules")

# ── LLM-OCR Prompt (pure text extraction, NO interpretation) ─────────────────

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
- Transcribe EXACTLY what you see — do not infer, round, or modify values
- For axis values, preserve the exact format (e.g., "$1.2M", "50%", "2024Q1")
- Read Y-axis top to bottom, X-axis left to right
- If a region has no text, use empty string or empty list
- Do NOT add any analysis or commentary"""


# ── Sample selection (same as other scripts) ─────────────────────────────────

def select_samples_aligned_with_b() -> list[dict]:
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def img_to_b64(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode in ("CMYK", "RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def vlm_call(client, model: str, image_b64: str, system: str, prompt: str) -> str:
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


def parse_numbers(values: list) -> list[float]:
    """Extract numeric values from OCR text list (e.g., ['$1.2M', '50%', '100'])."""
    numbers = []
    for v in values:
        if not isinstance(v, str):
            if isinstance(v, (int, float)):
                numbers.append(float(v))
            continue
        # Remove currency, percent, commas, units
        cleaned = re.sub(r'[,$%°]', '', v.strip())
        cleaned = re.sub(r'[KkMmBb]$', '', cleaned)  # Remove K/M/B suffix
        try:
            numbers.append(float(cleaned))
        except ValueError:
            pass
    return numbers


# ── Rule checks ──────────────────────────────────────────────────────────────

def run_rule_checks(engine: RuleEngine, ocr_data: dict) -> list[str]:
    results = []

    y_values = parse_numbers(ocr_data.get("y_axis_values", []))
    x_values = parse_numbers(ocr_data.get("x_axis_values", []))
    right_values = parse_numbers(ocr_data.get("right_y_axis_values", []))
    # Use chart_type from OCR extraction Call 1 (if available from judge step)
    chart_type = ocr_data.get("chart_type", "unknown")

    if y_values:
        try:
            r = engine.run_check("truncated_axis", {"axis_values": y_values, "chart_type": chart_type})
            if r["is_truncated"]:
                results.append(f"truncated_axis: {r['explanation']}")
        except Exception:
            pass

        try:
            r = engine.run_check("broken_scale", {"axis_values": y_values})
            if r["is_broken"]:
                results.append(f"broken_scale: {r['explanation']}")
        except Exception:
            pass

        try:
            r = engine.run_check("inverted_axis", {"axis_values": y_values})
            if r["is_inverted"]:
                results.append(f"inverted_axis: {r['explanation']}")
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
            r = engine.run_check("dual_axis", {
                "left_axis_values": y_values, "right_axis_values": right_values})
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


# ── Build OCR text summary for VLM Call 2 ────────────────────────────────────

def build_ocr_summary(ocr_data: dict) -> str:
    """Format OCR results like traditional OCR output."""
    lines = []
    if ocr_data.get("title"):
        lines.append(f"Title: {ocr_data['title']}")
    if ocr_data.get("y_axis_label"):
        lines.append(f"Y-axis label: {ocr_data['y_axis_label']}")
    if ocr_data.get("y_axis_values"):
        lines.append(f"Y-axis values: {', '.join(str(v) for v in ocr_data['y_axis_values'])}")
    if ocr_data.get("x_axis_label"):
        lines.append(f"X-axis label: {ocr_data['x_axis_label']}")
    if ocr_data.get("x_axis_values"):
        lines.append(f"X-axis values: {', '.join(str(v) for v in ocr_data['x_axis_values'])}")
    if ocr_data.get("right_y_axis_label"):
        lines.append(f"Right Y-axis: {ocr_data['right_y_axis_label']}")
    if ocr_data.get("right_y_axis_values"):
        lines.append(f"Right Y-axis values: {', '.join(str(v) for v in ocr_data['right_y_axis_values'])}")
    if ocr_data.get("legend_items"):
        lines.append(f"Legend: {', '.join(ocr_data['legend_items'])}")
    if ocr_data.get("data_labels"):
        lines.append(f"Data labels: {', '.join(str(v) for v in ocr_data['data_labels'])}")
    if ocr_data.get("source_text"):
        lines.append(f"Source: {ocr_data['source_text']}")
    if ocr_data.get("other_text"):
        lines.append(f"Other: {', '.join(ocr_data['other_text'])}")
    return "\n".join(lines) if lines else "No text detected."


# ── Process one chart ────────────────────────────────────────────────────────

def process_one(client, model: str, engine: RuleEngine, sample: dict) -> dict:
    iid = sample["instance_id"]
    image_b64 = img_to_b64(sample["image_path"])
    start = time.time()

    # Call 1: Pure LLM-OCR
    raw_ocr = vlm_call(client, model, image_b64, OCR_SYSTEM, OCR_PROMPT)
    ocr_data = extract_json(raw_ocr) or {}
    t_ocr = time.time() - start

    # Rule checks on OCR-extracted numbers
    rule_results = run_rule_checks(engine, ocr_data)

    # Build OCR text summary
    ocr_summary = build_ocr_summary(ocr_data)
    y_values = parse_numbers(ocr_data.get("y_axis_values", []))
    ocr_axis_str = ", ".join(str(v) for v in y_values) if y_values else "No axis values extracted."

    # Call 2: Final judgment with OCR + rule evidence
    misleader_list = "\n".join(f"- {k}: {v}" for k, v in MISLEADER_DEFINITIONS.items())
    completeness_list = "\n".join(f"- {k}: {v}" for k, v in COMPLETENESS_CHECKS.items())

    judge_prompt = PIPELINE_PROMPT.format(
        chart_id=f"eval_{iid}",
        page=1,
        ocr_text=ocr_summary,
        ocr_axis=ocr_axis_str,
        rule_results="\n".join(rule_results) if rule_results else "No rule checks applicable.",
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
        "t_ocr_s": round(t_ocr, 1),
        "ocr_y_values": ocr_data.get("y_axis_values", []),
        "ocr_right_values": ocr_data.get("right_y_axis_values", []),
        "ocr_has_title": bool(ocr_data.get("title")),
        "ocr_has_legend": bool(ocr_data.get("legend_items")),
        "rule_results": rule_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-OCR + Rules Pipeline experiment")
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

    print(f"\nStarting {args.workers}-thread LLM-OCR+Rules pipeline...")
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
                condition="llm_ocr_rules",
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
