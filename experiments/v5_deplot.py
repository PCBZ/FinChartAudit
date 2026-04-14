"""V5: B's prompt + DePlot post-processing (OCR CLEAN veto + table rule additions).

Unique to V5: After VLM, runs DePlot table analysis to ADD missed detections
(tick intervals, binning, inverted axis, axis range) in addition to CLEAN veto.

Usage:
    python experiments/v5_deplot.py --phase 1   # Run DePlot extraction first
    python experiments/v5_deplot.py --phase 2   # Run VLM + post-processing
    python experiments/v5_deplot.py             # Both phases
"""

import argparse
import subprocess
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

OUT_DIR = Path("data/eval_results/v5_deplot")

# ── Prompt (same as V4) ───────────────────────────────────────────────────────

USER_PROMPT = f"""You are an expert in data visualization. Detect misleading elements in the chart image.

## Misleader Taxonomy
{TAXONOMY_BLOCK}

## Examples
{FEW_SHOT_EXAMPLES}

## Output
Respond with valid JSON only:
{OUTPUT_FORMAT}"""


# ── V5-specific post-processing (OCR CLEAN veto + DePlot rule additions) ──────


def apply_postprocessing(
    predicted: list[str],
    ocr_data: dict,
    deplot_data: dict,
) -> tuple[list[str], list[str]]:
    """OCR CLEAN veto + DePlot table rule additions."""
    from finchartaudit.tools.table_rules import analyze_deplot_table

    result, veto_log = apply_clean_veto(predicted, ocr_data)

    if deplot_data and "error" not in deplot_data:
        table_checks = analyze_deplot_table(deplot_data)

        additions = [
            (
                "inconsistent tick intervals",
                "tick_intervals",
                "max_deviation",
                0.3,
                "deviation",
            ),
            ("inconsistent binning size", "binning", "max_deviation", 0.4, "deviation"),
        ]
        for label, key, metric, threshold, log_name in additions:
            if label not in result:
                check = table_checks.get(key, {})
                if check.get("flagged") and check.get(metric, 0) > threshold:
                    result.append(label)
                    veto_log.append(f"ADD {key}: {log_name}={check[metric]:.1%}")

        for label, key in [
            ("inverted axis", "inverted_axis"),
            ("inappropriate axis range", "axis_range"),
        ]:
            if label not in result:
                check = table_checks.get(key, {})
                if check.get("flagged"):
                    result.append(label)
                    veto_log.append(
                        f"ADD {key}: {check.get('reason', check.get('range_ratio', ''))}"
                    )

    return result, veto_log


# ── Per-sample worker ─────────────────────────────────────────────────────────


def make_call_fn(client, config, ocr_cache, deplot_cache):
    def call_fn(sample):
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=config.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
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
                temperature=0.0,
            )
            elapsed = time.time() - start
            data = extract_json(response.choices[0].message.content or "")
            predicted = data.get("misleader_types", []) if data else []
            if isinstance(predicted, str):
                predicted = [predicted]
            predicted, pp_log = apply_postprocessing(
                predicted, ocr_cache.get(iid, {}), deplot_cache.get(iid, {})
            )
            return {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(elapsed, 1),
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


# ── Phase 1: DePlot extraction ────────────────────────────────────────────────


def run_phase1(samples: list[dict]) -> None:
    """Run DePlot extraction as a subprocess for each sample."""
    print(f"Phase 1: DePlot extraction for {len(samples)} samples...")
    for s in samples:
        subprocess.run(
            [sys.executable, "-m", "finchartaudit.tools.deplot", s["image_path"]],
            check=False,
        )


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="V5: B's prompt + DePlot post-processing"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help="1=DePlot only, 2=VLM only, omit=both",
    )
    args = parser.parse_args()

    from finchartaudit.config import get_config
    from openai import OpenAI

    get_config.cache_clear()
    config = get_config()

    samples = select_samples()

    if args.phase in (1, None):
        run_phase1(samples)
    if args.phase in (2, None):
        client = OpenAI(
            api_key=config.openrouter_api_key, base_url=config.openrouter_base_url
        )
        ocr_cache = load_ocr_cache()
        deplot_cache = load_deplot_cache(samples)
        run_experiment(
            call_fn=make_call_fn(client, config, ocr_cache, deplot_cache),
            out_dir=OUT_DIR,
            condition_name="v5_deplot",
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
