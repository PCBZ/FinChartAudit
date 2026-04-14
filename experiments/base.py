"""Shared base utilities for all experiment scripts."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from src.utils import img_to_b64 as _img_to_b64
from src.utils import parse_json

# Re-export parse_json under the alias used by experiment scripts
extract_json = parse_json

OCR_CACHE_DIR = Path("data/eval_results/pipeline_full/ocr_cache")
BASELINE_RESULTS = Path(
    os.getenv("FCA_BASELINE_RESULTS", "results/claude_vision_only.json")
)


# ── Image helper ──────────────────────────────────────────────────────────────


def img_to_b64(image_path: str) -> str:
    """Load image and return base64-encoded JPEG string."""
    _, b64 = _img_to_b64(image_path)
    return b64


# ── Sample selection ──────────────────────────────────────────────────────────


def select_samples() -> list[dict]:
    """Load and align samples against the baseline results file.

    Reads baseline predictions from FCA_BASELINE_RESULTS (default:
    results/claude_vision_only.json) and matches them to local Misviz
    instances by misleader type + chart type.
    """
    from data_tools.misviz.loader import MisvizLoader

    if not BASELINE_RESULTS.exists():
        raise FileNotFoundError(
            f"Baseline results not found at {BASELINE_RESULTS}. "
            "Set FCA_BASELINE_RESULTS env var to the correct path."
        )

    b_data = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    b_items = b_data["results"]

    loader = MisvizLoader()
    real_data = loader.load_real()

    # Build index: (misleader_set, chart_type_tuple) → [local_indices]
    content_to_local: dict = defaultdict(list)
    for i, d in enumerate(real_data):
        key = (
            frozenset(d.get("misleader", [])),
            tuple(sorted(d.get("chart_type", []))),
        )
        content_to_local[key].append(i)

    samples = []
    used: set[int] = set()
    for b_item in b_items:
        key = (
            frozenset(b_item["gt_misleaders"]),
            tuple(sorted(b_item.get("chart_type", []))),
        )
        candidates = [idx for idx in content_to_local.get(key, []) if idx not in used]
        if candidates:
            idx = candidates[0]
            used.add(idx)
            instance = loader.get_real_instance(idx)
            if Path(instance.image_path).exists():
                samples.append(
                    {
                        "idx": idx,
                        "instance_id": str(b_item["id"]),
                        "image_path": instance.image_path,
                        "ground_truth": instance.misleader,
                        "b_id": b_item["id"],
                    }
                )

    print(f"Matched {len(samples)}/271 of B's samples")
    return samples


# ── Cache loaders ─────────────────────────────────────────────────────────────


def load_ocr_cache(cache_dir: Path = OCR_CACHE_DIR) -> dict:
    """Load OCR results from disk cache. Returns empty dict if cache missing."""
    cache: dict = {}
    if not cache_dir.exists():
        print(
            f"WARNING: No OCR cache at {cache_dir}. "
            "Run full_pipeline.py --phase 1 first."
        )
        return cache
    for f in sorted(cache_dir.glob("*.json")):
        try:
            cache.update(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cache


def load_deplot_cache(samples: list[dict]) -> dict:
    """Load cached DePlot results for the given samples."""
    from finchartaudit.tools.deplot import DePlotTool

    deplot = DePlotTool(device="cpu")
    cache: dict = {}
    for s in samples:
        result = deplot._cache_load(s["image_path"])
        if result:
            cache[s["instance_id"]] = result
    return cache


# ── Veto helpers ──────────────────────────────────────────────────────────────


def apply_clean_veto(
    predicted: list[str],
    ocr_data: dict,
) -> tuple[list[str], list[str]]:
    """Veto predictions where reliable OCR rules report [CLEAN].

    Only removes false positives — never adds new predictions.
    Returns (filtered_predictions, veto_log).
    """
    if not ocr_data or "error" in ocr_data:
        return predicted, []

    axis_values: list = ocr_data.get("axis_values", [])
    right_axis_values: list = ocr_data.get("right_axis_values", [])
    rule_results: list = ocr_data.get("rule_results", [])

    vetoed: list[str] = []
    veto_log: list[str] = []

    for name in predicted:
        if name == "truncated axis" and axis_values:
            trunc_flagged = any(r.startswith("truncated_axis:") for r in rule_results)
            if not trunc_flagged and min(axis_values) <= 0:
                veto_log.append(
                    f"VETO truncated_axis: axis includes 0 (min={min(axis_values)})"
                )
                continue
        if name == "dual axis" and not right_axis_values:
            dual_flagged = any(r.startswith("dual_axis:") for r in rule_results)
            if not dual_flagged:
                veto_log.append("VETO dual_axis: no right Y-axis detected by OCR")
                continue
        vetoed.append(name)

    return vetoed, veto_log


# ── Experiment runner ─────────────────────────────────────────────────────────


def run_experiment(
    call_fn: Callable[[dict], dict],
    out_dir: Path,
    condition_name: str,
    workers: int = 8,
    timeout: int = 120,
    samples: list[dict] | None = None,
) -> None:
    """Run a VLM experiment with resume support, progress logging, and metrics.

    Args:
        call_fn: Per-sample function. Receives a sample dict, returns a result
                 dict with keys: instance_id, ground_truth, predicted, elapsed_s.
                 On error, include an 'error' key.
        out_dir:        Output directory (created if missing).
        condition_name: Identifier string passed to MisvizEvaluator.
        workers:        ThreadPoolExecutor concurrency.
        timeout:        Per-future timeout in seconds.
        samples:        Pre-loaded sample list. If None, select_samples() is called.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if samples is None:
        samples = select_samples()

    # Resume: skip already-completed samples
    results_file = out_dir / "raw_results.json"
    if results_file.exists():
        existing: list[dict] = json.loads(results_file.read_text(encoding="utf-8"))
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

    def _tracked_worker(sample: dict) -> None:
        nonlocal completed, errors, total_vetoes
        result = call_fn(sample)
        with lock:
            completed += 1
            results.append(result)
            total_vetoes += len(result.get("veto_log", []))
            if "error" in result:
                errors += 1
                print(
                    f"[{completed}/{len(samples)}] id={result['instance_id']} "
                    f"ERROR: {result['error'][:60]}"
                )
            else:
                veto_str = (
                    f" vetoed={result['veto_log']}" if result.get("veto_log") else ""
                )
                print(
                    f"[{completed}/{len(samples)}] id={result['instance_id']} "
                    f"gt={result.get('ground_truth')} -> {result.get('predicted')} "
                    f"({result.get('elapsed_s', 0):.1f}s){veto_str}"
                )
            if completed % 20 == 0:
                _save(results, out_dir)

    print(f"\nStarting {condition_name} ({workers} threads)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_tracked_worker, s) for s in remaining]
        for fut in as_completed(futures):
            try:
                fut.result(timeout=timeout)
            except Exception as e:
                print(f"Thread error: {e}")

    _save(results, out_dir)
    total_time = time.time() - t0

    # Metrics via MisvizEvaluator
    from data_tools.misviz.evaluator import MisvizEvaluator

    evaluator = MisvizEvaluator()
    for r in results:
        if "error" not in r:
            evaluator.add_prediction(
                instance_id=r["instance_id"],
                ground_truth=r["ground_truth"],
                predicted=r["predicted"],
                condition=condition_name,
                model="claude_haiku",
            )
    evaluator.print_summary()
    metrics = evaluator.compute_metrics()
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"\nDone: {completed} charts, {errors} errors, {total_vetoes} vetoes applied")
    print(f"Time: {total_time:.0f}s ({total_time / max(completed, 1):.1f}s/chart)")


# ── Save helper ───────────────────────────────────────────────────────────────


def _save(results: list[dict], out_dir: Path) -> None:
    (out_dir / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )


# ── Legacy helpers (kept for compute_metrics callers) ─────────────────────────


def content_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_rule_verdicts(ocr_result: dict) -> dict:
    verdicts = {}
    for rule, result in (ocr_result or {}).items():
        if result.get("reliable"):
            verdicts[rule] = "[CLEAN]" if not result.get("flagged") else "[FLAGGED]"
        else:
            verdicts[rule] = "[INFO]"
    return verdicts


def apply_rule_veto(
    predicted: list[str], verdicts: dict
) -> tuple[list[str], list[str]]:
    veto_log = []
    filtered = []
    for t in predicted:
        if verdicts.get(t) == "[CLEAN]":
            veto_log.append(f"RULE_VETO: {t} (rule says CLEAN)")
            continue
        filtered.append(t)
    return filtered, veto_log


def compute_metrics(results: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for r in results:
        gt = bool(r.get("label"))
        pred = len(r.get("predicted", [])) > 0
        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif gt and not pred:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(results) if results else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(results),
    }


def run_parallel(fn, items: list, workers: int = 8) -> list:
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"error": str(e)}
    return results
