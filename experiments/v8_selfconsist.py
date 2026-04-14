"""V8: V7 sequential re-ask + self-consistency ×N voting on re-ask calls.

Architecture:
  Call 1: General detection (same as V7).
  Calls 2+: For charts Call 1 reports <=1 types, ask each blind spot type
             individually — but each re-ask runs N times (default 3); majority
             vote (>= N//2+1 YES) is required to accept.
  Post: OCR CLEAN veto + DePlot axis_range.

Usage:
    python experiments/v8_selfconsist.py
    python experiments/v8_selfconsist.py --workers 8 --votes 5
"""

import argparse
import json
import sys
import time
from collections import Counter
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

OUT_DIR = Path("data/eval_results/v8_selfconsist")

# ── Prompts ───────────────────────────────────────────────────────────────────

CALL1_PROMPT = f"""You are an expert in data visualization. Detect misleading elements in the chart image.

## Misleader Taxonomy
{TAXONOMY_BLOCK}

## Examples
{FEW_SHOT_EXAMPLES}

## Output
Respond with valid JSON only:
{OUTPUT_FORMAT}"""

# Per-type targeted prompts — more verbose than V7 to aid self-consistency
TARGETED_PROMPTS = {
    "inconsistent tick intervals": (
        "Look at the axis tick marks in this chart. "
        "Read the tick values along the Y-axis (or X-axis if more prominent). "
        "Are the intervals between consecutive tick values UNEVEN? "
        "For example: 0, 10, 20, 50, 100 has uneven gaps (10, 10, 30, 50). "
        'Does this chart have "inconsistent tick intervals"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "inconsistent binning size": (
        "Is this a histogram (bars representing numeric ranges)? "
        "If yes, look at the width of each bar carefully. "
        "Are the bars DIFFERENT widths? For example, one bar covers 0-10 while another covers 10-30. "
        'Does this chart have "inconsistent binning size"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "inverted axis": (
        "Look at the Y-axis numbers in this chart. "
        "Read them from bottom to top. Do they DECREASE (e.g., bottom=100, top=0)? "
        "That would mean the axis is inverted — high values at bottom, low at top. "
        'Does this chart have an "inverted axis"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "inappropriate axis range": (
        "Look at the Y-axis range in this chart. "
        "Does it show only a very narrow slice of values, making tiny differences look huge? "
        "For example: showing 98% to 102% instead of 0% to 100%, or 4.0 to 4.5 instead of 0 to 5. "
        'Does this chart have an "inappropriate axis range"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "discretized continuous variable": (
        "Look at this chart. Is the data inherently continuous (like temperature, time, money) "
        "but displayed in discrete bins or categories that hide the true distribution? "
        'For example: showing exact ages as "20-30, 30-40" ranges when finer bins would be better. '
        'Does this chart have a "discretized continuous variable"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
    "inappropriate item order": (
        "Look at how items are ordered in this chart. "
        "Are they arranged in a way that creates a false visual trend? "
        "For example: sorting countries by value to make it look like a declining trend "
        "when the items have no natural sequence. "
        'Does this chart have "inappropriate item order"? '
        'Answer with JSON: {"answer": "YES" or "NO", "reason": "brief explanation"}'
    ),
}


# ── V8-specific DePlot post-processing ───────────────────────────────────────


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


def make_call_fn(client, config, ocr_cache, deplot_cache, n_votes: int, threshold: int):
    def vote_reask(image_b64: str, prompt_text: str) -> tuple[bool, int]:
        """Run prompt N times at temperature=0.7; return (majority_yes, yes_count)."""
        yes_count = 0
        for _ in range(n_votes):
            try:
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
                    temperature=0.7,  # variation needed for self-consistency
                )
                data = extract_json(resp.choices[0].message.content or "")
                if (
                    data
                    and isinstance(data.get("answer"), str)
                    and data["answer"].upper().startswith("YES")
                ):
                    yes_count += 1
            except Exception:
                pass
        return yes_count >= threshold, yes_count

    def call_fn(sample):
        iid = sample["instance_id"]
        image_b64 = img_to_b64(sample["image_path"])
        try:
            start = time.time()
            veto_log = []
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

            # Sequential re-ask with self-consistency voting (only for clean/sparse)
            if len(predicted) <= 1:
                for label, prompt_text in TARGETED_PROMPTS.items():
                    if label in predicted:
                        continue
                    majority_yes, yes_count = vote_reask(image_b64, prompt_text)
                    n_calls += n_votes
                    if majority_yes:
                        predicted.append(label)
                        veto_log.append(f"VOTE_ADD {label}: {yes_count}/{n_votes}")
                    elif yes_count > 0:
                        veto_log.append(f"VOTE_REJECT {label}: {yes_count}/{n_votes}")

            # Post-processing: OCR CLEAN veto
            predicted, clean_log = apply_clean_veto(predicted, ocr_cache.get(iid, {}))
            veto_log.extend(clean_log)

            # Post-processing: DePlot axis_range
            predicted, deplot_log = apply_deplot_axis_range(
                predicted, deplot_cache.get(iid, {})
            )
            veto_log.extend(deplot_log)

            return {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(time.time() - start, 1),
                "n_calls": n_calls,
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
    parser = argparse.ArgumentParser(
        description="V8: sequential re-ask + self-consistency voting"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--votes",
        type=int,
        default=3,
        help="Votes per re-ask (default 3); majority = votes//2+1",
    )
    args = parser.parse_args()

    n_votes = args.votes
    threshold = n_votes // 2 + 1

    from finchartaudit.config import get_config
    from openai import OpenAI

    get_config.cache_clear()
    config = get_config()
    print(f"Self-consistency: {n_votes} votes, threshold >= {threshold}")

    client = OpenAI(
        api_key=config.openrouter_api_key, base_url=config.openrouter_base_url
    )
    samples = select_samples()
    ocr_cache = load_ocr_cache()
    deplot_cache = load_deplot_cache(samples)

    run_experiment(
        call_fn=make_call_fn(
            client, config, ocr_cache, deplot_cache, n_votes, threshold
        ),
        out_dir=OUT_DIR,
        condition_name="v8_selfconsist",
        workers=args.workers,
        timeout=600,
    )

    # Vote statistics from saved results
    results_file = OUT_DIR / "raw_results.json"
    if results_file.exists():
        results = json.loads(results_file.read_text(encoding="utf-8"))
        vote_adds = Counter()
        vote_rejects = Counter()
        for r in results:
            for log in r.get("veto_log", []):
                if log.startswith("VOTE_ADD "):
                    vote_adds[log.split(":")[0].removeprefix("VOTE_ADD ")] += 1
                elif log.startswith("VOTE_REJECT "):
                    vote_rejects[log.split(":")[0].removeprefix("VOTE_REJECT ")] += 1
        print(f"\nVote ADDs:    {dict(vote_adds.most_common())}")
        print(f"Vote REJECTs: {dict(vote_rejects.most_common())}")


if __name__ == "__main__":
    main()
