"""V7: B's prompt + sequential per-type re-ask + CLEAN veto + DePlot axis_range.

Key insight: VLM CAN detect blind spot types when asked individually.
The general 12-type prompt dilutes attention. Solution: for charts Call 1
reports as clean/sparse (<=1 types), ask each blind spot type ONE AT A TIME.

Usage:
    python experiments/v7_sequential.py
    python experiments/v7_sequential.py --workers 8
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.base import (
    apply_clean_veto,
    extract_json,
    img_to_b64,
    load_deplot_cache,
    load_ocr_cache,
    run_experiment,
    select_samples,
)
from src.prompts import FEW_SHOT_EXAMPLES, OUTPUT_FORMAT, TAXONOMY_BLOCK

OUT_DIR = Path("data/eval_results/v7_sequential")

# ── Prompts ───────────────────────────────────────────────────────────────────

CALL1_PROMPT = f"""You are an expert in data visualization. Detect misleading elements in the chart image.

## Misleader Taxonomy
{TAXONOMY_BLOCK}

## Examples
{FEW_SHOT_EXAMPLES}

## Output
Respond with valid JSON only:
{OUTPUT_FORMAT}"""

# One targeted YES/NO prompt per blind-spot type
TARGETED_PROMPTS = {
    "inconsistent tick intervals": (
        "Are the Y-axis tick intervals UNEVEN (e.g., 0, 10, 20, 50, 100 has gaps 10, 10, 30, 50)? "
        'Answer: {"answer": "YES/NO", "reason": "..."}'
    ),
    "inconsistent binning size": (
        "If this is a histogram, do bars have DIFFERENT widths? "
        'Answer: {"answer": "YES/NO", "reason": "..."}'
    ),
    "inverted axis": (
        "Do Y-axis numbers DECREASE from bottom to top (e.g., bottom=100, top=0)? "
        'Answer: {"answer": "YES/NO", "reason": "..."}'
    ),
    "inappropriate axis range": (
        "Does the Y-axis show only a very narrow slice making tiny differences look huge? "
        'Answer: {"answer": "YES/NO", "reason": "..."}'
    ),
    "discretized continuous variable": (
        "Is continuous data (temperature, money) binned into categories that hide its distribution? "
        'Answer: {"answer": "YES/NO", "reason": "..."}'
    ),
    "inappropriate item order": (
        "Are items ordered in a way that creates a false visual trend? "
        'Answer: {"answer": "YES/NO", "reason": "..."}'
    ),
}


# ── V7-specific DePlot post-processing ───────────────────────────────────────


def apply_deplot_axis_range(
    predicted: list[str], deplot_data: dict
) -> tuple[list[str], list[str]]:
    if not deplot_data or "error" in deplot_data:
        return predicted, []
    from finchartaudit.tools.table_rules import check_inappropriate_axis_range

    rows = deplot_data.get("rows", [])
    if not rows:
        return predicted, []
    check = check_inappropriate_axis_range(rows)
    if check.get("flagged") and "inappropriate axis range" not in predicted:
        return predicted + ["inappropriate axis range"], ["DEPLOT_ADD axis_range"]
    return predicted, []


# ── Per-sample worker ─────────────────────────────────────────────────────────


def make_call_fn(client, config, ocr_cache, deplot_cache):
    def call_fn(sample):
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])
        try:
            start = time.time()
            pp_log = []
            n_calls = 1

            # Call 1: general detection
            resp1 = client.chat.completions.create(
                model=config.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CALL1_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=512,
            )
            data1 = extract_json(resp1.choices[0].message.content or "")
            predicted = data1.get("misleader_types", []) if data1 else []
            if isinstance(predicted, str):
                predicted = [predicted]

            # Sequential re-ask: only for clean/sparse charts (<=1 types found)
            if len(predicted) <= 1:
                for label, prompt_text in TARGETED_PROMPTS.items():
                    if label in predicted:
                        continue
                    n_calls += 1
                    resp = client.chat.completions.create(
                        model=config.vlm_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_b64}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=150,
                    )
                    data = extract_json(resp.choices[0].message.content or "")
                    if (
                        data
                        and isinstance(data.get("answer"), str)
                        and data["answer"].upper().startswith("YES")
                    ):
                        predicted.append(label)
                        pp_log.append(f"SEQ_ADD {label}: {data.get('reason','')[:60]}")

            # Post-processing: OCR CLEAN veto
            predicted, veto_log = apply_clean_veto(predicted, ocr_cache.get(iid, {}))
            pp_log.extend(veto_log)

            # Post-processing: DePlot axis_range
            predicted, deplot_log = apply_deplot_axis_range(
                predicted, deplot_cache.get(iid, {})
            )
            pp_log.extend(deplot_log)

            return {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(time.time() - start, 1),
                "n_calls": n_calls,
                "veto_log": pp_log,
            }
        except Exception as e:
            return {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": [],
                "error": str(e),
            }

    return call_fn


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="V7: sequential per-type re-ask")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from finchartaudit.config import get_config
    from openai import OpenAI

    get_config.cache_clear()
    config = get_config()
    client = OpenAI(
        api_key=config.openrouter_api_key, base_url=config.openrouter_base_url
    )
    samples = select_samples()
    ocr_cache = load_ocr_cache()
    deplot_cache = load_deplot_cache(samples)

    run_experiment(
        call_fn=make_call_fn(client, config, ocr_cache, deplot_cache),
        out_dir=OUT_DIR,
        condition_name="v7_sequential",
        workers=args.workers,
        timeout=300,
    )


if __name__ == "__main__":
    main()
