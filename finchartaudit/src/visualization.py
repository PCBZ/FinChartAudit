# src/visualize.py
#
# Generate figures from aggregated results.
# Usage: python src/visualize.py
# Output: results/figures/

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "claude_vision_only": "Claude\nvision_only",
    "claude_vision_text": "Claude\nvision_text",
    "qwen_vision_only":   "Qwen\nvision_only",
    "qwen_vision_text":   "Qwen\nvision_text",
}

COLORS = {
    "claude_vision_only": "#185FA5",
    "claude_vision_text": "#85B7EB",
    "qwen_vision_only":   "#0F6E56",
    "qwen_vision_text":   "#5DCAA5",
}


def load_summary() -> dict:
    path = RESULTS_DIR / "aggregated_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Run run_pipeline.py first: {path}")
    return json.loads(path.read_text())


# ── Fig 1: RQ1/RQ2 2×2 heatmap ───────────────────────────────────────────────

def fig_2x2_heatmap(misviz_results: list[dict]):
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    keys = ["claude_vision_only", "claude_vision_text", "qwen_vision_only", "qwen_vision_text"]
    data = {}
    for r in misviz_results:
        key = f"{r['label'].split('|')[0].strip().lower()}_{r['label'].split('|')[1].strip().lower().replace(' ', '_')}"
        data[key] = r

    matrix = np.array([
        [data.get(k, {}).get(m, 0) for m in metrics]
        for k in keys
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([LABELS[k] for k in keys], fontsize=9)

    for i in range(len(keys)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "#222"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("RQ1/RQ2 — Misviz Benchmark", fontsize=12, pad=10)
    plt.tight_layout()
    path = FIGURES_DIR / "fig1_rq1_rq2_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path}")


# ── Fig 2: Per misleader type F1 bar chart ───────────────────────────────────

def fig_misleader_type(misviz_results: list[dict]):
    keys = ["claude_vision_only", "claude_vision_text", "qwen_vision_only", "qwen_vision_text"]
    data = {}
    for r in misviz_results:
        key = f"{r['label'].split('|')[0].strip().lower()}_{r['label'].split('|')[1].strip().lower().replace(' ', '_')}"
        data[key] = r.get("per_misleader_type_f1", {})

    all_types = sorted(set(t for d in data.values() for t in d.keys()))

    x = np.arange(len(all_types))
    n = len(keys)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, k in enumerate(keys):
        vals = [data.get(k, {}).get(t, 0) for t in all_types]
        ax.bar(x + i * width - (n - 1) * width / 2, vals,
               width=width, label=LABELS[k].replace("\n", " "),
               color=COLORS[k], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1 score", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("RQ1/RQ2 — Per Misleader Type F1", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "fig2_misleader_type_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path}")


# ── Fig 3: RQ3 per ticker heatmap ────────────────────────────────────────────

def fig_rq3_ticker(sec_results: list[dict]):
    keys = ["claude_vision_only", "claude_vision_text", "qwen_vision_only", "qwen_vision_text"]
    data = {}
    for r in sec_results:
        key = f"{r['label'].split('|')[0].strip().lower()}_{r['label'].split('|')[1].strip().lower().replace(' ', '_')}"
        data[key] = r.get("per_ticker", {})

    all_tickers = sorted(set(t for d in data.values() for t in d.keys()))

    matrix = np.array([
        [data.get(k, {}).get(ticker, {}).get("flag_rate", 0) for k in keys]
        for ticker in all_tickers
    ])

    fig, ax = plt.subplots(figsize=(7, len(all_tickers) * 0.55 + 1.5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([LABELS[k] for k in keys], fontsize=8)
    ax.set_yticks(range(len(all_tickers)))
    ax.set_yticklabels(all_tickers, fontsize=9)

    for i, ticker in enumerate(all_tickers):
        for j, k in enumerate(keys):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "#222"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Flag rate")
    ax.set_title("RQ3 — SEC Filings Flag Rate per Ticker", fontsize=11, pad=10)
    plt.tight_layout()
    path = FIGURES_DIR / "fig3_rq3_ticker_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path}")


# ── Fig 4: RQ3 overall metrics bar chart ─────────────────────────────────────

def fig_rq3_metrics(sec_results: list[dict]):
    keys = ["claude_vision_only", "claude_vision_text", "qwen_vision_only", "qwen_vision_text"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    data = {}
    for r in sec_results:
        key = f"{r['label'].split('|')[0].strip().lower()}_{r['label'].split('|')[1].strip().lower().replace(' ', '_')}"
        data[key] = r

    x = np.arange(len(metrics))
    n = len(keys)
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, k in enumerate(keys):
        vals = [data.get(k, {}).get(m, 0) for m in metrics]
        ax.bar(x + i * width - (n - 1) * width / 2, vals,
               width=width, label=LABELS[k].replace("\n", " "),
               color=COLORS[k], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("RQ3 — SEC Filings Overall Metrics", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "fig4_rq3_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    summary = load_summary()
    misviz  = summary.get("rq1_rq2_misviz", [])
    sec     = summary.get("rq3_sec", [])

    misviz = [r for r in misviz if r]
    sec    = [r for r in sec if r]

    if misviz:
        fig_2x2_heatmap(misviz)
        fig_misleader_type(misviz)
    else:
        print("⚠ No Misviz results found")

    if sec:
        fig_rq3_ticker(sec)
        fig_rq3_metrics(sec)
    else:
        print("⚠ No SEC results found")

    print(f"\n📊 Figures saved → {FIGURES_DIR}")


if __name__ == "__main__":
    main()