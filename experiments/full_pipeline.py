"""Two-phase Pipeline experiment on Misviz real dataset (aligned with B's 271 samples).

Phase 1: Batch OCR all images (subprocess per batch, solves memory leak).
Phase 2: Multi-threaded VLM calls reading pre-computed OCR (V3 tiered verdicts + rule veto).

Usage:
    python experiments/full_pipeline.py            # Full run (both phases)
    python experiments/full_pipeline.py --phase 1  # OCR only
    python experiments/full_pipeline.py --phase 2  # VLM only (requires Phase 1 done)
    python experiments/full_pipeline.py --workers 4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from experiments.base import (
    run_experiment, select_samples, load_ocr_cache,
    apply_clean_veto, img_to_b64, extract_json, _save,
)

PYTHON = sys.executable
OUT_DIR = Path("data/eval_results/pipeline_full")
OCR_CACHE_DIR = OUT_DIR / "ocr_cache"
BATCH_SIZE = 5  # charts per OCR subprocess


# ── Phase 1: Batch OCR ────────────────────────────────────────────────────────

OCR_WORKER_SCRIPT = OUT_DIR / "_ocr_worker.py"


def write_ocr_worker() -> None:
    """Write a worker script that OCRs a batch of images."""
    OCR_WORKER_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    project_root = str(Path(__file__).parent.resolve()).replace("\\", "\\\\")
    worker_code = '''
"""Worker: OCR a batch of images. Input: JSON list of {id, image_path} on stdin."""
import json, os, sys, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
sys.path.insert(0, r"__PROJECT_ROOT__")

batch = json.loads(sys.stdin.read())

from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
from finchartaudit.tools.rule_check import RuleEngine
import re

ocr = TraditionalOCRTool()
engine = RuleEngine()
results = {}


def extract_numbers(result):
    numbers = []
    for b in result.get("text_blocks", result.get("texts", [])):
        text = b.get("text", b) if isinstance(b, dict) else str(b)
        for match in re.findall(r"-?\\d+\\.?\\d*", text):
            try:
                numbers.append(float(match))
            except ValueError:
                pass
    return sorted(set(numbers))


def format_ocr(result):
    lines = [
        b.get("text", "").strip()
        for b in result.get("text_blocks", [])[:20]
        if isinstance(b, dict) and b.get("confidence", 0) > 0.5 and b.get("text", "").strip()
    ]
    return "\\n".join(lines) if lines else "No confident text detected."


for item in batch:
    image_path = item["image_path"]
    iid = item["id"]
    try:
        full_result  = ocr.run(image_path, "full",       "bbox")
        y_result     = ocr.run(image_path, "y_axis",     "text")
        right_result = ocr.run(image_path, "right_axis", "text")
        x_result     = ocr.run(image_path, "x_axis",     "text")

        axis_values       = extract_numbers(y_result)
        right_axis_values = extract_numbers(right_result)
        x_axis_values     = extract_numbers(x_result)

        rule_results = []
        if axis_values:
            for check_type in ["truncated_axis", "broken_scale"]:
                try:
                    r = engine.run_check(check_type, {"axis_values": axis_values, "chart_type": "bar"})
                    rule_results.append(f"{check_type}: {r['explanation']}")
                except Exception:
                    pass
            for check_type, flag_key in [("inverted_axis", "is_inverted"),
                                          ("inappropriate_axis_range", "is_inappropriate")]:
                try:
                    r = engine.run_check(check_type, {"axis_values": axis_values})
                    if r[flag_key]:
                        rule_results.append(f"{check_type}: {r['explanation']}")
                except Exception:
                    pass

        if axis_values and right_axis_values:
            try:
                r = engine.run_check("dual_axis", {
                    "left_axis_values": axis_values, "right_axis_values": right_axis_values})
                if r["has_dual_axis"]:
                    rule_results.append(f"dual_axis: {r['explanation']}")
            except Exception:
                pass

        if len(x_axis_values) >= 3:
            try:
                r = engine.run_check("inconsistent_binning", {"bin_edges": x_axis_values})
                if r["is_inconsistent"]:
                    rule_results.append(f"inconsistent_binning: {r['explanation']}")
            except Exception:
                pass

        results[iid] = {
            "ocr_text":          format_ocr(full_result),
            "ocr_axis":          format_ocr(y_result),
            "axis_values":       axis_values,
            "right_axis_values": right_axis_values,
            "x_axis_values":     x_axis_values,
            "rule_results":      rule_results,
        }
    except Exception as e:
        results[iid] = {"error": str(e)}

print(json.dumps(results))
'''.replace("__PROJECT_ROOT__", project_root)
    OCR_WORKER_SCRIPT.write_text(worker_code, encoding="utf-8")


def run_phase1(samples: list[dict]) -> None:
    """Batch OCR all images using subprocess isolation."""
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_ocr_worker()

    done: set[str] = set()
    for f in OCR_CACHE_DIR.glob("*.json"):
        try:
            done.update(json.loads(f.read_text(encoding="utf-8")).keys())
        except Exception:
            pass

    remaining = [s for s in samples if s["instance_id"] not in done]
    print(f"Phase 1: OCR {len(remaining)} images ({len(done)} cached)")
    if not remaining:
        print("All OCR already cached, skipping Phase 1")
        return

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} images)...", end="", flush=True)
        batch_input = [{"id": s["instance_id"], "image_path": s["image_path"]} for s in batch]
        t = time.time()
        try:
            result = subprocess.run(
                [PYTHON, "-X", "utf8", str(OCR_WORKER_SCRIPT)],
                input=json.dumps(batch_input),
                capture_output=True, text=True, timeout=900,
                env={**os.environ, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
            )
            if result.returncode != 0:
                print(f" ERROR: {result.stderr[-200:]}")
                continue
            batch_results = json.loads(result.stdout)
            cache_file = OCR_CACHE_DIR / f"batch_{batch_num:04d}.json"
            cache_file.write_text(json.dumps(batch_results, indent=2), encoding="utf-8")
            errors = sum(1 for v in batch_results.values() if "error" in v)
            print(f" done ({time.time() - t:.0f}s, {errors} errors)")
        except subprocess.TimeoutExpired:
            print(" TIMEOUT")
        except Exception as e:
            print(f" ERROR: {e}")

    if OCR_WORKER_SCRIPT.exists():
        OCR_WORKER_SCRIPT.unlink()


# ── Phase 2 helpers ───────────────────────────────────────────────────────────

def build_rule_verdicts(
    axis_values: list, right_axis_values: list,
    x_axis_values: list, raw_rule_results: list[str],
) -> str:
    """Build tiered rule verdicts string for the VLM prompt.

    RELIABLE rules → [CLEAN]/[FLAGGED] (VLM must respect).
    UNRELIABLE rules → [INFO] (VLM decides on its own).
    """
    if not axis_values:
        return ("No numeric Y-axis values extracted by OCR. "
                "Rule checks could not run.\nUse your visual analysis only.")

    lines = []
    trunc_flagged = any("instead of 0" in r.lower() or "exaggerated" in r.lower()
                        for r in raw_rule_results if r.startswith("truncated_axis:"))
    if trunc_flagged:
        lines.append(f"[FLAGGED] truncated_axis: Y-axis starts at {min(axis_values)}, not 0.")
    elif min(axis_values) <= 0:
        lines.append(f"[CLEAN] truncated_axis: Y-axis includes 0 (min={min(axis_values)}). NOT truncated.")
    else:
        lines.append(f"[CLEAN] truncated_axis: Y-axis min={min(axis_values)}. Rule did not flag.")

    dual_flagged = any(r.startswith("dual_axis:") for r in raw_rule_results)
    if dual_flagged:
        lines.append("[FLAGGED] dual_axis: Left and right Y-axes detected with different scales.")
    elif right_axis_values:
        lines.append(f"[INFO] dual_axis: Right Y-axis values found: {right_axis_values[:6]}. Verify visually.")
    else:
        lines.append("[CLEAN] dual_axis: No right Y-axis detected by OCR.")

    inv_flagged = any(r.startswith("inverted_axis:") for r in raw_rule_results)
    lines.append(
        f"[INFO] inverted_axis: OCR reads values top-to-bottom as increasing ({axis_values[:4]}...). "
        f"{'MAY indicate inverted axis.' if inv_flagged else 'Use image to verify axis direction.'}"
    )

    iar_flagged = any(r.startswith("inappropriate_axis_range:") for r in raw_rule_results)
    val_range = max(axis_values) - min(axis_values)
    lines.append(
        f"[INFO] inappropriate_axis_range: Range {min(axis_values)}-{max(axis_values)} "
        f"(span={val_range:.1f}). "
        f"{'Flagged as narrow — is this a bar/area chart?' if iar_flagged else 'Judge from image.'}"
    )

    broken_flagged = any("inconsistent" in r.lower() and r.startswith("broken_scale:")
                         for r in raw_rule_results)
    lines.append(
        f"[INFO] inconsistent_tick_intervals: Values {axis_values[:8]}. "
        f"{'Rule detected uneven spacing. Verify visually.' if broken_flagged else 'Check image for even tick spacing.'}"
    )

    if any(r.startswith("inconsistent_binning:") for r in raw_rule_results):
        lines.append("[INFO] inconsistent_binning: X-axis bin widths appear unequal.")

    return "\n".join(lines)


def _parse_predicted(text: str) -> list[str]:
    """Extract predicted types from a misleaders-dict VLM response."""
    data = extract_json(text)
    if not data:
        return []
    return list({
        name for name, assessment in data.get("misleaders", {}).items()
        if isinstance(assessment, dict)
        and assessment.get("present")
        and float(assessment.get("confidence", 0)) >= 0.3
    })


# ── Phase 2: VLM worker ───────────────────────────────────────────────────────

def make_phase2_call_fn(client, config, ocr_cache):
    from finchartaudit.agents.t2_pipeline import PIPELINE_SYSTEM_PROMPT, PIPELINE_PROMPT
    from finchartaudit.prompts.t2_visual import COMPLETENESS_CHECKS

    completeness_list = "\n".join(f"- {k}: {v}" for k, v in COMPLETENESS_CHECKS.items())

    def call_fn(sample):
        iid = sample["instance_id"]
        ocr_data = ocr_cache.get(iid, {})

        if "error" in ocr_data or not ocr_data:
            ocr_axis, ocr_x_str, rule_verdicts = (
                "OCR failed.", "OCR failed.", "No rule checks (OCR failed).")
            axis_values = right_axis_values = rule_results = []
        else:
            axis_values       = ocr_data.get("axis_values", [])
            right_axis_values = ocr_data.get("right_axis_values", [])
            x_axis_values     = ocr_data.get("x_axis_values", [])
            rule_results      = ocr_data.get("rule_results", [])
            ocr_axis          = ocr_data.get("ocr_axis", "No axis values.")
            ocr_x_str         = (", ".join(str(v) for v in x_axis_values[:15])
                                 if x_axis_values else "Not extracted")
            rule_verdicts     = build_rule_verdicts(
                axis_values, right_axis_values, x_axis_values, rule_results)

        prompt = PIPELINE_PROMPT.format(
            chart_id=f"eval_{iid}", page=1,
            ocr_axis=ocr_axis, ocr_x_axis=ocr_x_str,
            rule_verdicts=rule_verdicts, completeness_list=completeness_list,
        )

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=config.vlm_model,
                messages=[
                    {"role": "system", "content": PIPELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{img_to_b64(sample['image_path'])}"}},
                    ]},
                ],
                max_tokens=2048, temperature=0.0,
            )
            predicted = _parse_predicted(response.choices[0].message.content or "")
            # OCR CLEAN veto (truncated_axis + dual_axis)
            predicted, veto_log = apply_clean_veto(predicted, ocr_data)
            return {
                "instance_id": iid, "ground_truth": sample["ground_truth"],
                "predicted": predicted, "elapsed_s": round(time.time() - start, 1),
                "veto_log": veto_log,
            }
        except Exception as e:
            return {"instance_id": iid, "ground_truth": sample["ground_truth"],
                    "predicted": [], "error": str(e)}
    return call_fn


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full Pipeline: batch OCR + VLM")
    parser.add_argument("--phase", type=int, default=0,
                        choices=[0, 1, 2], help="0=both, 1=OCR only, 2=VLM only")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Selecting samples...")
    samples = select_samples()
    print(f"Selected {len(samples)} samples")

    if args.phase in (0, 1):
        print(f"\n{'='*60}\nPHASE 1: Batch OCR\n{'='*60}")
        t1 = time.time()
        run_phase1(samples)
        print(f"Phase 1 completed in {time.time() - t1:.0f}s")

    if args.phase in (0, 2):
        print(f"\n{'='*60}\nPHASE 2: Multi-threaded VLM\n{'='*60}")
        from finchartaudit.config import get_config
        from openai import OpenAI
        get_config.cache_clear()
        config = get_config()
        client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)
        ocr_cache = load_ocr_cache(OCR_CACHE_DIR)
        print(f"OCR cache: {len(ocr_cache)} entries")
        missing = [s for s in samples if s["instance_id"] not in ocr_cache]
        if missing:
            print(f"WARNING: {len(missing)} samples missing OCR — run --phase 1 first")

        run_experiment(
            call_fn=make_phase2_call_fn(client, config, ocr_cache),
            out_dir=OUT_DIR,
            condition_name="pipeline_full",
            workers=args.workers,
            timeout=120,
            samples=samples,
        )


if __name__ == "__main__":
    main()
