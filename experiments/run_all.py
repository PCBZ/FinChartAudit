"""Run all three pending experiments sequentially.

1. T3 Case Study (7 case companies) — ~5-10 min
2. False Positive Test (3 clean companies) — ~3-5 min
3. Tool-use Ablation (100 Misviz real charts) — ~60-90 min
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
sys.path.insert(0, str(Path(__file__).parent))

from finchartaudit.config import get_config
from finchartaudit.vlm.claude_client import OpenRouterVLMClient
from finchartaudit.memory.filing_memory import FilingMemory
from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
from finchartaudit.agents.orchestrator import Orchestrator
from finchartaudit.agents.t2_visual import T2VisualAgent
from data_tools.misviz.loader import MisvizLoader
from data_tools.misviz.evaluator import MisvizEvaluator

# Clear config cache to pick up new .env
get_config.cache_clear()
config = get_config()

FILINGS_DIR = Path("data/filings")
EVAL_DIR = Path("data/eval_results")

# Company groups from SEC信息汇总.xlsx
CASE_COMPANIES = ["ALV", "MYE", "FXLV", "UIS", "OC", "CNM", "AAP"]
CLEAN_COMPANIES = ["CTAS", "SHW", "ROK"]


def run_t3_on_companies(companies: list[str], experiment_name: str):
    """Run T3 pairing analysis on a list of companies."""
    print(f"\n{'#'*70}")
    print(f"# Experiment: {experiment_name}")
    print(f"# Companies: {companies}")
    print(f"{'#'*70}")

    vlm = OpenRouterVLMClient(api_key=config.openrouter_api_key, model=config.vlm_model)
    ocr = TraditionalOCRTool()
    results = {}

    for company in companies:
        filing_dir = FILINGS_DIR / company / "filing"
        if not filing_dir.exists():
            print(f"  [SKIP] {company}: no filing directory")
            continue

        filings = [f for f in filing_dir.iterdir()
                   if f.suffix in (".htm", ".html", ".pdf") and "_meta" not in f.name]
        if not filings:
            print(f"  [SKIP] {company}: no filing files")
            continue

        for filing_path in filings:
            print(f"\n{'='*60}")
            print(f"[{company}] {filing_path.name}")
            print(f"{'='*60}")

            memory = FilingMemory()
            orchestrator = Orchestrator(vlm=vlm, memory=memory)
            orchestrator.set_ocr_tool(ocr)

            try:
                start = time.time()
                findings = orchestrator.audit_filing(
                    file_path=str(filing_path),
                    ticker=company,
                    filing_type=filing_path.stem.split("_")[-1] if "_" in filing_path.stem else "",
                )
                elapsed = time.time() - start

                print(f"  Findings: {len(findings)} ({elapsed:.1f}s)")
                for f in findings:
                    print(f"    [{f.risk_level}] {f.subcategory}: {f.description[:80]}")

                results[f"{company}/{filing_path.name}"] = {
                    "ticker": company,
                    "file": filing_path.name,
                    "findings_count": len(findings),
                    "elapsed_s": round(elapsed, 1),
                    "findings": [f.to_dict() for f in findings],
                    "summary": memory.get_summary(),
                    "pairing_count": len(memory.pairing_matrix),
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[f"{company}/{filing_path.name}"] = {
                    "ticker": company, "file": filing_path.name, "error": str(e),
                }

            time.sleep(1)

    # Save results
    out_dir = EVAL_DIR / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "results.json"
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary: {experiment_name}")
    print(f"{'='*60}")
    print(f"{'Ticker':<8} {'File':<35} {'Findings':>8} {'Pairings':>9} {'Time':>6}")
    print("-" * 70)
    total_findings = 0
    for key, data in results.items():
        if "error" not in data:
            total_findings += data["findings_count"]
            print(f"{data['ticker']:<8} {data['file']:<35} {data['findings_count']:>8} "
                  f"{data.get('pairing_count', 'N/A'):>9} {data['elapsed_s']:>5.1f}s")
        else:
            print(f"{data['ticker']:<8} {data['file']:<35} {'ERROR':>8}")
    print(f"\nTotal findings: {total_findings}")
    print(f"Results saved to: {output}")
    return results


def run_tooluse_ablation(n: int = 100):
    """Run tool-use ablation on Misviz real dataset.

    Compares our VLM+OCR+Rules pipeline against B's VLM-only baseline.
    """
    print(f"\n{'#'*70}")
    print(f"# Experiment: Tool-Use Ablation (n={n})")
    print(f"# Model: {config.vlm_model}")
    print(f"# Pipeline: VLM + PaddleOCR + RuleEngine (tool-use loop)")
    print(f"{'#'*70}")

    vlm = OpenRouterVLMClient(api_key=config.openrouter_api_key, model=config.vlm_model)
    ocr = TraditionalOCRTool()
    loader = MisvizLoader()
    evaluator = MisvizEvaluator()

    # Load real dataset (same as B used)
    real_data = loader.load_real()

    # Stratified sample similar to B's approach: sample across misleader types
    from collections import defaultdict
    type_buckets = defaultdict(list)
    clean_indices = []

    for i, d in enumerate(real_data):
        misleaders = d.get("misleader", [])
        if not misleaders:
            clean_indices.append(i)
        else:
            for m in misleaders:
                type_buckets[m].append(i)

    # Take up to 8 per type + 30 clean
    selected = set()
    for mtype, indices in type_buckets.items():
        selected.update(indices[:8])
    selected.update(clean_indices[:30])
    selected = sorted(selected)[:n]

    print(f"Selected {len(selected)} instances (stratified)")

    out_dir = EVAL_DIR / "tooluse_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = 0
    results = []

    for idx in selected:
        instance = loader.get_real_instance(idx)
        if not Path(instance.image_path).exists():
            continue

        count += 1
        print(f"[{count}/{len(selected)}] id={instance.instance_id} gt={instance.misleader}", end="", flush=True)

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
                "tool_calls": sum(1 for f in findings for _ in f.tool_calls) if findings else 0,
            })
            print(f" -> {predicted} ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            print(f" -> ERROR: {e}")
            results.append({
                "instance_id": instance.instance_id,
                "error": str(e),
            })

        time.sleep(0.3)

        # Save intermediate results every 20 charts
        if count % 20 == 0:
            _save_ablation_results(out_dir, results, evaluator, count, errors)

    # Final save
    _save_ablation_results(out_dir, results, evaluator, count, errors)
    evaluator.print_summary()

    print(f"\nDone: {count} charts, {errors} errors")
    return results


def _save_ablation_results(out_dir, results, evaluator, count, errors):
    (out_dir / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    metrics = evaluator.compute_metrics()
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  [saved] {count} results, {errors} errors")


if __name__ == "__main__":
    total_start = time.time()

    # Experiment 1: T3 Case Study (quick)
    case_results = run_t3_on_companies(CASE_COMPANIES, "t3_case_study")

    # Experiment 2: False Positive Test (quick)
    clean_results = run_t3_on_companies(CLEAN_COMPANIES, "t3_false_positive")

    # Experiment 3: Tool-use Ablation (long)
    ablation_results = run_tooluse_ablation(n=100)

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*70}")
    print(f"# ALL EXPERIMENTS COMPLETE ({total_elapsed/60:.1f} min)")
    print(f"{'#'*70}")
