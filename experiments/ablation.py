"""Pipeline ablation — runs T2 Pipeline agent on Misviz real dataset.

Uses subprocess per chart to avoid PaddleOCR memory leaks.

Usage:
    python run_pipeline_ablation.py          # 50 charts
    python run_pipeline_ablation.py --n 20   # quick test
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PYTHON = sys.executable
OUT_DIR = Path("data/eval_results/pipeline_ablation")
WORKER_SCRIPT = Path("_pipeline_worker.py")


def select_samples(n: int) -> list[dict]:
    """Stratified sample from Misviz real dataset."""
    from data_tools.misviz.loader import MisvizLoader
    loader = MisvizLoader()
    real_data = loader.load_real()

    type_buckets = defaultdict(list)
    clean_indices = []
    for i, d in enumerate(real_data):
        misleaders = d.get("misleader", [])
        if not misleaders:
            clean_indices.append(i)
        else:
            for m in misleaders:
                type_buckets[m].append(i)

    selected = set()
    for mtype, indices in type_buckets.items():
        selected.update(indices[:5])
    selected.update(clean_indices[:15])
    selected = sorted(selected)[:n]

    samples = []
    for idx in selected:
        instance = loader.get_real_instance(idx)
        samples.append({
            "idx": idx,
            "instance_id": str(idx),
            "image_path": instance.image_path,
            "ground_truth": instance.misleader,
        })
    return samples


def run_single_chart(sample: dict) -> dict:
    """Run pipeline agent on one chart in a subprocess."""
    result = subprocess.run(
        [PYTHON, "-X", "utf8", str(WORKER_SCRIPT), sample["image_path"]],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
    )
    if result.returncode != 0:
        return {
            "instance_id": sample["instance_id"],
            "ground_truth": sample["ground_truth"],
            "predicted": [],
            "error": result.stderr[-500:] if result.stderr else "unknown error",
        }

    try:
        output = json.loads(result.stdout)
        return {
            "instance_id": sample["instance_id"],
            "ground_truth": sample["ground_truth"],
            "predicted": output.get("predicted", []),
            "findings_count": output.get("findings_count", 0),
            "elapsed_s": output.get("elapsed_s", 0),
            "ocr_text_len": output.get("ocr_text_len", 0),
            "axis_values_count": output.get("axis_values_count", 0),
        }
    except json.JSONDecodeError:
        return {
            "instance_id": sample["instance_id"],
            "ground_truth": sample["ground_truth"],
            "predicted": [],
            "error": f"JSON parse error: {result.stdout[-200:]}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    # Write the worker script
    write_worker_script()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Selecting {args.n} stratified samples...")
    samples = select_samples(args.n)
    print(f"Selected {len(samples)} samples")

    # Check for existing results to resume
    results_file = OUT_DIR / "raw_results.json"
    if results_file.exists():
        existing = json.loads(results_file.read_text(encoding="utf-8"))
        done_ids = {r["instance_id"] for r in existing if "error" not in r}
        print(f"Resuming: {len(done_ids)} already done")
    else:
        existing = []
        done_ids = set()

    results = list(existing)
    errors = sum(1 for r in existing if "error" in r)
    count = len(existing)

    for sample in samples:
        if sample["instance_id"] in done_ids:
            continue

        count += 1
        gt = sample["ground_truth"]
        print(f"[{count}/{len(samples)}] id={sample['instance_id']} gt={gt}", end="", flush=True)

        try:
            result = run_single_chart(sample)
        except subprocess.TimeoutExpired:
            result = {
                "instance_id": sample["instance_id"],
                "ground_truth": gt,
                "predicted": [],
                "error": "timeout",
            }

        if "error" in result:
            errors += 1
            print(f" -> ERROR: {result['error'][:80]}")
        else:
            print(f" -> {result['predicted']} ({result.get('elapsed_s', '?')}s)")

        results.append(result)

        # Save every 5
        if count % 5 == 0:
            _save(results)

    _save(results)

    # Compute metrics
    from data_tools.misviz.evaluator import MisvizEvaluator
    ev = MisvizEvaluator()
    for r in results:
        if "error" not in r:
            ev.add_prediction(
                r["instance_id"], r["ground_truth"], r["predicted"],
                condition="pipeline", model="claude_haiku")

    ev.print_summary()
    metrics = ev.compute_metrics()
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nDone: {count} charts, {errors} errors")
    print(f"Results: {OUT_DIR}")

    # Cleanup
    if WORKER_SCRIPT.exists():
        WORKER_SCRIPT.unlink()


def _save(results):
    (OUT_DIR / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")


def write_worker_script():
    """Write a self-contained worker that processes one chart."""
    WORKER_SCRIPT.write_text('''
"""Worker: process one chart with T2 Pipeline agent. Outputs JSON to stdout."""
import json
import os
import sys
import time

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Suppress all non-essential output to stderr
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

image_path = sys.argv[1]

from finchartaudit.config import get_config
get_config.cache_clear()
config = get_config()

from finchartaudit.vlm.claude_client import OpenRouterVLMClient
from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
from finchartaudit.agents.t2_pipeline import T2PipelineAgent

vlm = OpenRouterVLMClient(api_key=config.openrouter_api_key, model=config.vlm_model)
memory = FilingMemory()

# OCR init
try:
    ocr = TraditionalOCRTool()
    has_ocr = ocr.is_available
except Exception:
    ocr = None
    has_ocr = False

agent = T2PipelineAgent(vlm=vlm, memory=memory)
if has_ocr:
    agent.set_ocr_tool(ocr)

start = time.time()
findings = agent.execute({"image_path": image_path, "page": 1, "chart_id": "eval"})
elapsed = time.time() - start

predicted = list({f.subcategory for f in findings if f.category == "misleader"})

output = {
    "predicted": predicted,
    "findings_count": len(findings),
    "elapsed_s": round(elapsed, 1),
    "ocr_text_len": sum(len(b.get("text","")) for e in memory.audit_trace.get_trace()
                        for b in [] if e.tool_name == "traditional_ocr"),
    "axis_values_count": 0,
}

print(json.dumps(output))
''', encoding="utf-8")


if __name__ == "__main__":
    main()
