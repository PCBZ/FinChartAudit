"""Shared evaluation metrics for Misviz and SEC experiments.

Extracted from run_pipeline.py so experiment scripts can import
instead of re-implementing metric calculations.
"""
from __future__ import annotations
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def is_positive(pred: dict) -> bool:
    return bool(pred.get("pred_misleading") or pred.get("pred_violation"))


def aggregate_misviz(result: dict, label: str) -> dict:
    rows = [r for r in result.get("results", []) if r]
    if not rows:
        return {}
    y_true = [1 if len(r.get("gt_misleaders", [])) > 0 else 0 for r in rows]
    y_pred = [1 if r.get("pred_misleading") else 0 for r in rows]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for r in rows:
        gt_types = set(r.get("gt_misleaders", []))
        pred_types = set(r.get("pred_misleader_types", []))
        for t in gt_types | pred_types:
            if t in gt_types and t in pred_types: by_type[t]["tp"] += 1
            elif t in pred_types: by_type[t]["fp"] += 1
            elif t in gt_types: by_type[t]["fn"] += 1
    type_f1 = {}
    for t, c in by_type.items():
        yt = [1]*(c["tp"]+c["fn"]) + [0]*(c["fp"]+c["tn"])
        yp = [1]*c["tp"] + [0]*c["fn"] + [1]*c["fp"] + [0]*c["tn"]
        type_f1[t] = round(f1_score(yt, yp, zero_division=0), 3)
    return {"label": label, "total": len(rows),
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "per_misleader_type_f1": dict(sorted(type_f1.items(), key=lambda x: x[1], reverse=True))}


def aggregate_sec(result: dict, label: str) -> dict:
    all_preds = [item for items in result.get("results", {}).values() for item in items]
    if not all_preds:
        return {}
    y_true = [1 if p.get("has_gt_violation") else 0 for p in all_preds]
    y_pred = [1 if is_positive(p) else 0 for p in all_preds]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    per_ticker = {}
    for ticker, items in result.get("results", {}).items():
        if not items: continue
        flagged = sum(1 for p in items if is_positive(p))
        per_ticker[ticker] = {"total": len(items), "flagged": flagged,
                              "flag_rate": round(flagged/len(items), 3),
                              "has_gt_violation": items[0].get("has_gt_violation", False)}
    return {"label": label, "model": result.get("model",""), "condition": result.get("condition",""),
            "total": len(all_preds),
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "per_ticker": per_ticker}


def print_table(results: list[dict], title: str):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")
    print(f"{'Label':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}\n{'-'*65}")
    for r in results:
        if r: print(f"{r['label']:<40} {r['accuracy']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
