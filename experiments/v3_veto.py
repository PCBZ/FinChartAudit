"""V3: VLM-only detection + Rule [CLEAN] post-processing veto.

VLM sees ONLY the image (no OCR data in prompt).
After VLM returns predictions, apply rule veto using cached OCR/rule data:
  - truncated_axis [CLEAN] → veto VLM's "truncated axis" prediction
  - dual_axis [CLEAN] → veto VLM's "dual axis" prediction

Usage:
    python experiments/v3_veto.py
    python experiments/v3_veto.py --workers 4
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
    load_ocr_cache,
    run_experiment,
)

OUT_DIR = Path("data/eval_results/vlm_only_veto")

# ── Prompts ───────────────────────────────────────────────────────────────────

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


# ── Per-sample worker ─────────────────────────────────────────────────────────


def make_call_fn(client, config, ocr_cache):
    def call_fn(sample):
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=config.vlm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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
                    },
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            elapsed = time.time() - start
            data = extract_json(response.choices[0].message.content or "")
            predicted = data.get("misleader_types", []) if data else []
            if isinstance(predicted, str):
                predicted = [predicted]
            predicted, veto_log = apply_clean_veto(predicted, ocr_cache.get(iid, {}))
            return {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(elapsed, 1),
                "veto_log": veto_log,
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
    parser = argparse.ArgumentParser(description="V3: VLM-only + CLEAN veto")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from finchartaudit.config import get_config
    from openai import OpenAI

    get_config.cache_clear()
    config = get_config()
    client = OpenAI(
        api_key=config.openrouter_api_key, base_url=config.openrouter_base_url
    )
    ocr_cache = load_ocr_cache()

    run_experiment(
        call_fn=make_call_fn(client, config, ocr_cache),
        out_dir=OUT_DIR,
        condition_name="vlm_only_veto",
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
