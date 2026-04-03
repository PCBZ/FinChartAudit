"""Batch evaluation runner for T2 agent on Misviz dataset.

Usage:
    python -m finchartaudit.eval.run_t2_batch --n 50 --dataset synth --output data/eval_results/t2_pilot
    python -m finchartaudit.eval.run_t2_batch --n 1000 --dataset synth --condition vision_only --model claude
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from finchartaudit.config import get_config
from finchartaudit.vlm.claude_client import OpenRouterVLMClient
from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
from finchartaudit.agents.t2_visual import T2VisualAgent
from data_tools.misviz.loader import MisvizLoader
from data_tools.misviz.evaluator import MisvizEvaluator


MISVIZ_TO_FCA = {
    "truncated axis": "truncated axis",
    "inverted axis": "inverted axis",
    "misrepresentation": "misrepresentation",
    "3d": "3d",
    "dual axis": "dual axis",
    "inappropriate use of pie chart": "inappropriate use of pie chart",
    "inappropriate use of line chart": "inappropriate use of line chart",
    "inconsistent binning size": "inconsistent binning size",
    "inconsistent tick intervals": "inconsistent tick intervals",
    "discretized continuous variable": "discretized continuous variable",
    "inappropriate item order": "inappropriate item order",
    "inappropriate axis range": "inappropriate axis range",
}


def run_batch(n: int, dataset: str, output_dir: str,
              condition: str = "vision_only", model_name: str = "claude"):
    config = get_config()
    model_id = config.vlm_model if model_name == "claude" else config.qwen_model

    vlm = OpenRouterVLMClient(api_key=config.openrouter_api_key, model=model_id)
    ocr = TraditionalOCRTool()
    loader = MisvizLoader()
    evaluator = MisvizEvaluator()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if dataset == "synth":
        raw_data = loader.load_synth()
    else:
        raw_data = loader.load_real()

    count = 0
    errors = 0
    results = []

    for idx, raw in enumerate(raw_data):
        if count >= n:
            break

        if dataset == "synth":
            instance = loader.get_synth_instance(idx)
        else:
            instance = loader.get_real_instance(idx)

        if not Path(instance.image_path).exists():
            continue

        count += 1
        print(f"[{count}/{n}] id={instance.instance_id} gt={instance.misleader}")

        memory = FilingMemory()
        agent = T2VisualAgent(vlm=vlm, memory=memory)
        agent.set_ocr_tool(ocr)

        try:
            start = time.time()
            findings = agent.execute({
                "image_path": instance.image_path,
                "page": 1,
                "chart_id": f"eval_{instance.instance_id}",
            })
            elapsed = time.time() - start

            predicted = list({f.subcategory for f in findings if f.category == "misleader"})
            gt = [MISVIZ_TO_FCA.get(m, m) for m in instance.misleader]

            evaluator.add_prediction(
                instance_id=instance.instance_id,
                ground_truth=gt,
                predicted=predicted,
                confidences={f.subcategory: f.confidence for f in findings if f.category == "misleader"},
                condition=condition,
                model=model_name,
            )

            result_entry = {
                "instance_id": instance.instance_id,
                "ground_truth": gt,
                "predicted": predicted,
                "findings_count": len(findings),
                "elapsed_s": round(elapsed, 1),
                "all_findings": [f.to_dict() for f in findings],
            }
            results.append(result_entry)
            print(f"  -> predicted={predicted} ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            print(f"  -> ERROR: {e}")
            results.append({
                "instance_id": instance.instance_id,
                "error": str(e),
            })

        time.sleep(0.5)

    results_file = out / "raw_results.json"
    results_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    experiment_name = f"t2_{dataset}_{model_name}_{condition}_n{n}"
    metrics = evaluator.save_results(experiment_name)
    evaluator.print_summary()

    print(f"\nDone: {count} charts, {errors} errors")
    print(f"Results: {results_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run T2 batch evaluation on Misviz")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--dataset", choices=["synth", "real"], default="synth")
    parser.add_argument("--output", type=str, default="data/eval_results/t2_pilot")
    parser.add_argument("--condition", choices=["vision_only", "vision_text"], default="vision_only")
    parser.add_argument("--model", choices=["claude", "qwen"], default="claude")
    args = parser.parse_args()

    run_batch(args.n, args.dataset, args.output, args.condition, args.model)


if __name__ == "__main__":
    main()
