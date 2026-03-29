# src/run_pipeline.py
#
# FinChartAudit — End-to-End Pipeline
# From raw data → experiments → aggregation → report
#
# Usage:
#   python src/run_pipeline.py
#   python src/run_pipeline.py --workers 8

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
api_key = os.environ["OPENROUTER_API_KEY"]

from src.eval_runner import evaluate, evaluate_sec
from src.data.extract_charts import extract_all as extract_charts
from src.data.extract_tables import extract_all as extract_tables
from src.data.extract_nongaap import extract_all as extract_nongaap
from src.data.download_sec_data import main as download_sec_data
from src.data.download_letter_text import download_all_letters


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Data preparation
# ══════════════════════════════════════════════════════════════════════════════

TICKERS = [
    "TSLA", "LYFT", "ABNB", "RIVN", "VERI", "CHPT",
    "UPS", "KMI", "OGN", "BCO", "STZ", "IRBT", "LITE",
]


def _missing_tickers(directory: str, suffix: str = ".json") -> list:
    d = Path(directory)
    if not d.exists():
        return TICKERS
    existing = {p.stem.upper() for p in d.glob(f"*{suffix}")}
    return [t for t in TICKERS if t not in existing]


def _missing_letter_tickers() -> list:
    d = Path("data/letters")
    if not d.exists():
        return TICKERS
    existing = {p.name.upper() for p in d.iterdir() if p.is_dir() and any(p.iterdir())}
    return [t for t in TICKERS if t not in existing]


def prepare_data():
    print("\n" + "="*60)
    print("  STEP 0 — Data Preparation")
    print("="*60)

    missing = _missing_tickers("data/sec")
    if missing:
        print(f"\n[SEC] Missing {len(missing)} tickers: {missing}")
        download_sec_data()
    else:
        print("\n[SEC] All tickers found, skipping download")

    missing = _missing_letter_tickers()
    if missing:
        print(f"\n[Letters] Missing {len(missing)} tickers: {missing}")
        download_all_letters()
    else:
        print("\n[Letters] All tickers found, skipping download")

    missing = _missing_tickers("data/charts")
    if missing:
        print(f"\n[Charts] Missing {len(missing)} tickers: {missing}")
        extract_charts()
    else:
        print("\n[Charts] All tickers found, skipping extraction")

    missing = _missing_tickers("data/tables")
    if missing:
        print(f"\n[Tables] Missing {len(missing)} tickers: {missing}")
        extract_tables()
    else:
        print("\n[Tables] All tickers found, skipping extraction")

    if not Path("data/ground_truth.json").exists():
        print("\n[Ground Truth] Extracting Non-GAAP mentions from comment letters...")
        extract_nongaap()
    else:
        print("\n[Ground Truth] ground_truth.json found, skipping extraction")

    print("\n✅ Data preparation complete")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Run all experiments
# ══════════════════════════════════════════════════════════════════════════════

def _is_valid_result(path: Path, error_threshold: float = 0.1) -> bool:
    """Check if a result file exists and has acceptable error rate."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        # Misviz: results is a list
        # SEC: results is a dict of ticker -> list
        results = data.get("results", [])
        if isinstance(results, dict):
            rows = [item for items in results.values() for item in items if item]
        else:
            rows = [r for r in results if r]
        if not rows:
            return False
        error_rate = sum(1 for r in rows if r.get("api_error")) / len(rows)
        return error_rate <= error_threshold
    except Exception:
        return False


def run_all_experiments(workers: int, sample: int = None):
    for model_key in ("claude", "qwen"):
        for condition in ("vision_only", "vision_text"):

            print(f"\n{'#'*60}")
            print(f"  [Misviz] {model_key.upper()} | {condition}")
            print(f"{'#'*60}")
            evaluate(
                api_key=api_key,
                model_key=model_key,
                condition=condition,
                workers=workers,
                n_samples=sample,
            )

            print(f"\n{'#'*60}")
            print(f"  [SEC] {model_key.upper()} | {condition}")
            print(f"{'#'*60}")
            evaluate_sec(
                api_key=api_key,
                model_key=model_key,
                condition=condition,
                workers=workers,
                max_per_ticker=sample if sample else 10,
            )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Aggregate & report
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"⚠ File not found: {path}")
        return {}
    return json.loads(p.read_text())


def is_positive(pred: dict) -> bool:
    return bool(pred.get("pred_misleading") or pred.get("pred_violation"))


def aggregate_misviz(result: dict, label: str) -> dict:
    rows = result.get("results", [])
    rows = [r for r in rows if r]
    if not rows:
        return {}

    y_true = [1 if len(r.get("gt_misleaders", [])) > 0 else 0 for r in rows]
    y_pred = [1 if r.get("pred_misleading") else 0 for r in rows]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for r in rows:
        gt_types   = set(r.get("gt_misleaders", []))
        pred_types = set(r.get("pred_misleader_types", []))
        for t in gt_types | pred_types:
            gt   = t in gt_types
            pred = t in pred_types
            if pred and gt:           by_type[t]["tp"] += 1
            elif pred and not gt:     by_type[t]["fp"] += 1
            elif not pred and gt:     by_type[t]["fn"] += 1
            else:                     by_type[t]["tn"] += 1

    type_f1 = {}
    for t, c in by_type.items():
        yt = [1] * (c["tp"] + c["fn"]) + [0] * (c["fp"] + c["tn"])
        yp = [1] * c["tp"] + [0] * c["fn"] + [1] * c["fp"] + [0] * c["tn"]
        type_f1[t] = round(f1_score(yt, yp, zero_division=0), 3)

    return {
        "label":     label,
        "total":     len(rows),
        "accuracy":  round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "per_misleader_type_f1": dict(sorted(type_f1.items(), key=lambda x: x[1], reverse=True)),
    }


def aggregate_sec(result: dict, label: str) -> dict:
    all_preds = [
        item
        for items in result.get("results", {}).values()
        for item in items
    ]
    if not all_preds:
        return {}

    y_true = [1 if p.get("has_gt_violation") else 0 for p in all_preds]
    y_pred = [1 if is_positive(p) else 0 for p in all_preds]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    per_ticker = {}
    for ticker, items in result.get("results", {}).items():
        if not items:
            continue
        flagged  = sum(1 for p in items if is_positive(p))
        has_gt_t = items[0].get("has_gt_violation", False)
        per_ticker[ticker] = {
            "total":            len(items),
            "flagged":          flagged,
            "flag_rate":        round(flagged / len(items), 3) if items else 0,
            "has_gt_violation": has_gt_t,
        }

    return {
        "label":     label,
        "model":     result.get("model", ""),
        "condition": result.get("condition", ""),
        "total":     len(all_preds),
        "accuracy":  round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "per_ticker": per_ticker,
    }


def print_table(results: list[dict], title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"{'Label':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"{'-'*65}")
    for r in results:
        if not r:
            continue
        print(f"{r['label']:<40} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f}")


def print_misleader_breakdown(results: list[dict]):
    print(f"\n{'='*65}")
    print("  RQ1/RQ2 — Per Misleader Type F1")
    print(f"{'='*65}")
    all_types = set()
    for r in results:
        all_types.update(r.get("per_misleader_type_f1", {}).keys())
    col_w = 12
    print(f"{'Misleader Type':<30}" + "".join(f"{r['label'][:col_w]:>{col_w}}" for r in results))
    print("-" * (30 + col_w * len(results)))
    for t in sorted(all_types):
        row = f"{t:<30}"
        for r in results:
            val = r.get("per_misleader_type_f1", {}).get(t, "-")
            row += f"{str(val):>{col_w}}"
        print(row)


def print_sec_per_ticker(results: list[dict]):
    print(f"\n{'='*65}")
    print("  RQ3 — Per Ticker Detail")
    print(f"{'='*65}")
    all_tickers = set()
    for r in results:
        all_tickers.update(r.get("per_ticker", {}).keys())
    col_w = 16
    print(f"{'Ticker':<8} {'GT':>4}" + "".join(f"{r['label'][:col_w]:>{col_w}}" for r in results))
    print("-" * (12 + col_w * len(results)))
    for ticker in sorted(all_tickers):
        gt_mark = " "
        cells = ""
        for r in results:
            info = r.get("per_ticker", {}).get(ticker)
            if info:
                gt_mark = "✓" if info["has_gt_violation"] else " "
                cells += f"  {info['flagged']}/{info['total']}({info['flag_rate']:.0%}){'':<4}"
            else:
                cells += f"{'N/A':>{col_w}}"
        print(f"{ticker:<8} {gt_mark:>4}  {cells}")


def run_aggregation():
    results_dir = Path("results")

    misviz_files = {
        "Claude | vision_only": "claude_vision_only.json",
        "Claude | vision_text": "claude_vision_text.json",
        "Qwen   | vision_only": "qwen_vision_only.json",
        "Qwen   | vision_text": "qwen_vision_text.json",
    }
    sec_files = {
        "Claude | vision_only": "sec_claude_vision_only.json",
        "Claude | vision_text": "sec_claude_vision_text.json",
        "Qwen   | vision_only": "sec_qwen_vision_only.json",
        "Qwen   | vision_text": "sec_qwen_vision_text.json",
    }

    misviz_results = []
    for label, fname in misviz_files.items():
        data = load_json(results_dir / fname)
        if data:
            misviz_results.append(aggregate_misviz(data, label))

    sec_results = []
    for label, fname in sec_files.items():
        data = load_json(results_dir / fname)
        if data:
            sec_results.append(aggregate_sec(data, label))

    misviz_results = [r for r in misviz_results if r]
    sec_results    = [r for r in sec_results if r]

    if misviz_results:
        print_table(misviz_results, "RQ1/RQ2 — Misviz Benchmark")
        print_misleader_breakdown(misviz_results)

    if sec_results:
        print_table(sec_results, "RQ3 — SEC Filings")
        print_sec_per_ticker(sec_results)

    summary = {"rq1_rq2_misviz": misviz_results, "rq3_sec": sec_results}
    out = results_dir / "aggregated_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Summary saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinChartAudit Pipeline")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of concurrent API threads (default: 16)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Quick test: n_samples for Misviz, max_per_ticker for SEC")
    args = parser.parse_args()

    print(f"🚀 Starting FinChartAudit pipeline with {args.workers} workers"
          + (f" | sample={args.sample}" if args.sample else ""))
    prepare_data()
    run_all_experiments(args.workers, args.sample)
    run_aggregation()