"""Shared base utilities for all experiment scripts."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.utils import parse_json


def content_hash(path: str | Path) -> str:
    """SHA256 of file content for cache keying."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_samples(dataset, n: int = 271, reference_hashes: set | None = None) -> list:
    """Select samples, optionally matched by content hash to a reference set."""
    if reference_hashes is None:
        return list(dataset)[:n]
    matched = [s for s in dataset if content_hash(s.get("image", "")) in reference_hashes]
    return matched[:n]


def build_rule_verdicts(ocr_result: dict) -> dict:
    """Build tiered rule verdicts: [CLEAN] / [FLAGGED] / [INFO]."""
    verdicts = {}
    for rule, result in (ocr_result or {}).items():
        if result.get("reliable"):
            verdicts[rule] = "[CLEAN]" if not result.get("flagged") else "[FLAGGED]"
        else:
            verdicts[rule] = "[INFO]"
    return verdicts


def apply_rule_veto(predicted: list[str], verdicts: dict) -> tuple[list[str], list[str]]:
    """Veto VLM predictions contradicted by reliable rules. Returns (filtered, veto_log)."""
    veto_log = []
    filtered = []
    for t in predicted:
        if verdicts.get(t) == "[CLEAN]":
            veto_log.append(f"RULE_VETO: {t} (rule says CLEAN)")
            continue
        filtered.append(t)
    return filtered, veto_log


def compute_metrics(results: list[dict]) -> dict:
    """Compute precision, recall, F1 from results. Each item needs 'label' and 'predicted'."""
    tp = fp = fn = tn = 0
    for r in results:
        gt = bool(r.get("label"))
        pred = len(r.get("predicted", [])) > 0
        if gt and pred:       tp += 1
        elif not gt and pred: fp += 1
        elif gt and not pred: fn += 1
        else:                 tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(results) if results else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": len(results),
    }


def run_parallel(fn, items: list, workers: int = 8) -> list:
    """Run fn(item) for each item in parallel. Returns results in original order."""
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
