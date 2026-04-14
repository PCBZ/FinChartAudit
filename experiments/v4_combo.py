"""V4: B's prompt + disambiguation few-shots + CLEAN veto.

Strategy:
  Use B's prompt structure with 6 few-shot examples targeting top FP sources.
  Post-processing: CLEAN veto from cached OCR/rule data.

Usage:
    python experiments/v4_combo.py
    python experiments/v4_combo.py --workers 4
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import TAXONOMY_BLOCK, FEW_SHOT_EXAMPLES, OUTPUT_FORMAT
from experiments.base import (
    run_experiment, load_ocr_cache, apply_clean_veto, img_to_b64, extract_json,
)

OUT_DIR = Path("data/eval_results/v4_combo")

# ── Prompt (B's structure: no system prompt, all in user message) ─────────────

USER_PROMPT = f"""You are an expert in data visualization. Detect misleading elements in the chart image.

## Misleader Taxonomy
{TAXONOMY_BLOCK}

## Examples
{FEW_SHOT_EXAMPLES}

## Output
Respond with valid JSON only:
{OUTPUT_FORMAT}"""


# ── Per-sample worker ─────────────────────────────────────────────────────────

def make_call_fn(client, config, ocr_cache):
    def call_fn(sample):
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
                "explanation": (data or {}).get("explanation", ""),
            }
        except Exception as e:
            return {"instance_id": iid, "ground_truth": sample["ground_truth"],
                    "predicted": [], "error": str(e)}
    return call_fn


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V4: B's prompt + CLEAN veto")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from finchartaudit.config import get_config
    from openai import OpenAI
    get_config.cache_clear()
    config = get_config()
    client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)
    ocr_cache = load_ocr_cache()

    run_experiment(
        call_fn=make_call_fn(client, config, ocr_cache),
        out_dir=OUT_DIR,
        condition_name="v4_combo",
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
