"""Run T3 pairing analysis on all SEC case study companies.

Usage:
    python -m finchartaudit.eval.run_t3_casestudy
    python -m finchartaudit.eval.run_t3_casestudy --ticker MYE
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
from finchartaudit.agents.orchestrator import Orchestrator


FILINGS_DIR = Path("data/filings")
EVAL_DIR = Path("data/eval_results/t3_casestudy")


def run_casestudy(ticker: str | None = None):
    config = get_config()
    vlm = OpenRouterVLMClient(api_key=config.openrouter_api_key, model=config.vlm_model)
    ocr = TraditionalOCRTool()

    results = {}

    companies = sorted([d.name for d in FILINGS_DIR.iterdir() if d.is_dir()])
    if ticker:
        companies = [c for c in companies if c == ticker]

    for company in companies:
        filing_dir = FILINGS_DIR / company / "filing"
        if not filing_dir.exists():
            continue

        filings = [f for f in filing_dir.iterdir()
                   if f.suffix in (".htm", ".html", ".pdf") and "_meta" not in f.name]
        if not filings:
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
                    file_path=str(filing_path), ticker=company,
                    filing_type=filing_path.stem.split("_")[-1] if "_" in filing_path.stem else "")
                elapsed = time.time() - start

                print(f"  Findings: {len(findings)}")
                for f in findings:
                    print(f"    [{f.risk_level}] {f.subcategory}: {f.description[:80]}")

                results[f"{company}/{filing_path.name}"] = {
                    "ticker": company,
                    "file": filing_path.name,
                    "findings_count": len(findings),
                    "elapsed_s": round(elapsed, 1),
                    "findings": [f.to_dict() for f in findings],
                    "summary": memory.get_summary(),
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"{company}/{filing_path.name}"] = {
                    "ticker": company, "file": filing_path.name, "error": str(e)}

            time.sleep(1)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output = EVAL_DIR / "casestudy_results.json"
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {output}")

    print(f"\n{'Ticker':<8} {'File':<30} {'Findings':>8} {'Time':>6}")
    print("-" * 56)
    for key, data in results.items():
        if "error" not in data:
            print(f"{data['ticker']:<8} {data['file']:<30} {data['findings_count']:>8} {data['elapsed_s']:>5.1f}s")
        else:
            print(f"{data['ticker']:<8} {data['file']:<30} {'ERROR':>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    run_casestudy(args.ticker)


if __name__ == "__main__":
    main()
