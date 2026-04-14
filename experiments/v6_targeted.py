"""V6: B's prompt + targeted blind-spot re-ask + CLEAN veto + DePlot axis_range.

Unique to V6: VLM Call 2 asks targeted YES/NO questions for blind spot types
(tick intervals, binning, inverted axis, axis range) ONLY for charts Call 1
reported as clean or lightly flagged.

Usage:
    python experiments/v6_targeted.py
    python experiments/v6_targeted.py --workers 4
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import TAXONOMY_BLOCK, FEW_SHOT_EXAMPLES, OUTPUT_FORMAT
from experiments.base import (
    run_experiment, select_samples, load_ocr_cache, load_deplot_cache,
    apply_clean_veto, img_to_b64, extract_json,
)

OUT_DIR = Path("data/eval_results/v6_targeted")

# ── Prompts ───────────────────────────────────────────────────────────────────

CALL1_PROMPT = f"""You are an expert in data visualization. Detect misleading elements in the chart image.

## Misleader Taxonomy
{TAXONOMY_BLOCK}

## Examples
{FEW_SHOT_EXAMPLES}

## Output
Respond with valid JSON only:
{OUTPUT_FORMAT}"""

CALL2_PROMPT = """Look at this chart image very carefully and answer each question with YES or NO.

Q1 - TICK INTERVALS: Are the Y-axis tick gaps UNEVEN (e.g., 0, 10, 20, 50, 100)?
Q2 - BINNING: If a histogram, do bars have DIFFERENT widths?
Q3 - INVERTED AXIS: Do Y-axis numbers run HIGH at bottom to LOW at top?
Q4 - AXIS RANGE: Does the Y-axis show only a very narrow slice making tiny differences look huge?

Respond with ONLY valid JSON:
{
  "tick_intervals": {"answer": "YES/NO", "reason": "..."},
  "binning":        {"answer": "YES/NO", "reason": "..."},
  "inverted_axis":  {"answer": "YES/NO", "reason": "..."},
  "axis_range":     {"answer": "YES/NO", "reason": "..."}
}"""

BLIND_SPOT_MAP = {
    "tick_intervals": "inconsistent tick intervals",
    "binning":        "inconsistent binning size",
    "inverted_axis":  "inverted axis",
    "axis_range":     "inappropriate axis range",
}


# ── V6-specific post-processing ───────────────────────────────────────────────

def apply_deplot_axis_range(
    predicted: list[str], deplot_data: dict
) -> tuple[list[str], list[str]]:
    """Add inappropriate axis range from DePlot table rules if VLM missed it."""
    if not deplot_data or "error" in deplot_data:
        return predicted, []
    from finchartaudit.tools.table_rules import check_inappropriate_axis_range
    rows = deplot_data.get("rows", [])
    if not rows:
        return predicted, []
    check = check_inappropriate_axis_range(rows)
    if check.get("flagged") and "inappropriate axis range" not in predicted:
        return predicted + ["inappropriate axis range"], \
               [f"ADD axis_range: {check.get('reason', '')}"]
    return predicted, []


# ── Per-sample worker ─────────────────────────────────────────────────────────

def make_call_fn(client, config, ocr_cache, deplot_cache):
    def call_fn(sample):
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])
        try:
            start = time.time()

            # Call 1: general detection
            resp1 = client.chat.completions.create(
                model=config.vlm_model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": CALL1_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]}],
                max_tokens=512,
            )
            data1 = extract_json(resp1.choices[0].message.content or "")
            predicted = data1.get("misleader_types", []) if data1 else []
            if isinstance(predicted, str):
                predicted = [predicted]

            # Call 2: targeted re-ask for blind spots not caught in Call 1
            pp_log = []
            missing = [bs for bs in BLIND_SPOT_MAP.values() if bs not in predicted]
            had_call2 = bool(missing)
            if missing:
                resp2 = client.chat.completions.create(
                    model=config.vlm_model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": CALL2_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ]}],
                    max_tokens=512,
                )
                data2 = extract_json(resp2.choices[0].message.content or "")
                if data2:
                    for key, label in BLIND_SPOT_MAP.items():
                        if label in predicted:
                            continue
                        ans = data2.get(key, {})
                        if isinstance(ans, dict) and ans.get("answer", "").upper().startswith("YES"):
                            predicted.append(label)
                            pp_log.append(f"CALL2_ADD {key}: {ans.get('reason','')[:60]}")

            # Post-processing: OCR CLEAN veto + DePlot axis_range
            predicted, veto_log = apply_clean_veto(predicted, ocr_cache.get(iid, {}))
            pp_log.extend(veto_log)
            predicted, deplot_log = apply_deplot_axis_range(predicted, deplot_cache.get(iid, {}))
            pp_log.extend(deplot_log)

            return {
                "instance_id": iid, "ground_truth": sample["ground_truth"],
                "predicted": predicted, "elapsed_s": round(time.time() - start, 1),
                "veto_log": pp_log, "had_call2": had_call2,
            }
        except Exception as e:
            return {"instance_id": iid, "ground_truth": sample["ground_truth"],
                    "predicted": [], "error": str(e)}
    return call_fn


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V6: targeted blind-spot re-ask")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from finchartaudit.config import get_config
    from openai import OpenAI
    get_config.cache_clear()
    config = get_config()
    client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)
    samples = select_samples()
    ocr_cache = load_ocr_cache()
    deplot_cache = load_deplot_cache(samples)

    run_experiment(
        call_fn=make_call_fn(client, config, ocr_cache, deplot_cache),
        out_dir=OUT_DIR,
        condition_name="v6_targeted",
        workers=args.workers,
        timeout=180,
    )


if __name__ == "__main__":
    main()
