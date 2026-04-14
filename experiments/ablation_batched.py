"""Tool-use ablation — memory-safe batched version.

Reuses OCR instance, creates fresh VLM client per batch to avoid memory buildup.
"""
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
sys.path.insert(0, str(Path(__file__).parent))

from finchartaudit.config import get_config
from finchartaudit.vlm.claude_client import OpenRouterVLMClient
from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
from finchartaudit.agents.t2_visual import T2VisualAgent
from data_tools.misviz.loader import MisvizLoader
from data_tools.misviz.evaluator import MisvizEvaluator

get_config.cache_clear()
config = get_config()
print(f"Model: {config.vlm_model}")

N = 50
BATCH_SIZE = 10
OUT_DIR = Path("data/eval_results/tooluse_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load existing results if resuming
results_file = OUT_DIR / "raw_results.json"
if results_file.exists():
    existing = json.loads(results_file.read_text(encoding="utf-8"))
    done_ids = {r["instance_id"] for r in existing if "error" not in r}
    print(f"Resuming: {len(existing)} already done")
else:
    existing = []
    done_ids = set()

# Init OCR once (expensive)
print("Initializing OCR...")
ocr = TraditionalOCRTool()

# Load dataset and select samples
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
selected = sorted(selected)[:N]

# Filter out already done
remaining = [idx for idx in selected if str(idx) not in done_ids]
print(f"Selected {len(selected)}, remaining {len(remaining)}")

results = list(existing)
evaluator = MisvizEvaluator()

# Re-add existing predictions to evaluator
for r in existing:
    if "error" not in r:
        evaluator.add_prediction(
            instance_id=r["instance_id"],
            ground_truth=r["ground_truth"],
            predicted=r["predicted"],
            condition="tooluse",
            model="claude_haiku",
        )

count = len(existing)
errors = sum(1 for r in existing if "error" in r)

# Process in batches
for batch_start in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[batch_start:batch_start + BATCH_SIZE]
    print(f"\n--- Batch {batch_start // BATCH_SIZE + 1} ({len(batch)} charts) ---")

    # Fresh VLM per batch to avoid conversation memory buildup
    vlm = OpenRouterVLMClient(api_key=config.openrouter_api_key, model=config.vlm_model)

    for idx in batch:
        instance = loader.get_real_instance(idx)
        if not Path(instance.image_path).exists():
            continue

        count += 1
        print(f"[{count}/{N}] id={instance.instance_id} gt={instance.misleader}", end="", flush=True)

        memory = FilingMemory()
        agent = T2VisualAgent(vlm=vlm, memory=memory)
        agent.set_ocr_tool(ocr)

        try:
            start = time.time()
            findings = agent.execute({
                "image_path": instance.image_path,
                "page": 1,
                "chart_id": f"ablation_{instance.instance_id}",
            })
            elapsed = time.time() - start

            predicted = list({f.subcategory for f in findings if f.category == "misleader"})
            gt = instance.misleader

            evaluator.add_prediction(
                instance_id=instance.instance_id,
                ground_truth=gt,
                predicted=predicted,
                confidences={f.subcategory: f.confidence for f in findings if f.category == "misleader"},
                condition="tooluse",
                model="claude_haiku",
            )

            results.append({
                "instance_id": instance.instance_id,
                "ground_truth": gt,
                "predicted": predicted,
                "findings_count": len(findings),
                "elapsed_s": round(elapsed, 1),
            })
            print(f" -> {predicted} ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            print(f" -> ERROR: {e}")
            results.append({"instance_id": instance.instance_id, "error": str(e)})

        # Clear memory
        del memory, agent
        time.sleep(0.3)

    # Save after each batch
    results_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    metrics = evaluator.compute_metrics()
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  [saved] {count} done, {errors} errors")

    # Force GC between batches
    del vlm
    gc.collect()

# Final summary
evaluator.print_summary()
print(f"\nDone: {count} charts, {errors} errors")
print(f"Results: {OUT_DIR}")
