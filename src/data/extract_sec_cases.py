"""Extract chart images from SEC filings and run T2 Pipeline + T3 analysis.

1. Parse HTML filings for <img> tags
2. Download images from SEC EDGAR
3. Filter: skip logos/icons (too small), keep chart-sized images
4. Run T2 Pipeline on each chart
5. Combine with T3 results for full audit report
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
sys.path.insert(0, str(Path(__file__).parent))

import httpx

FILINGS_DIR = Path("data/filings")
CHARTS_DIR = Path("data/charts")
EVAL_DIR = Path("data/eval_results/sec_full_audit")
PYTHON = sys.executable

HEADERS = {
    "User-Agent": "FinChartAudit Research project@cornell.edu",
    "Accept-Encoding": "gzip, deflate",
}

# Company metadata: ticker -> (cik, accession_number from filing URL)
# Extracted from the filing URLs in SEC信息汇总.xlsx
COMPANY_META = {
    "ALV": {"cik": "1034670", "accession": "000156459022002816"},
    "MYE": {"cik": "69488", "accession": "000095017023005324"},
    "FXLV": {"cik": "1788717", "accession": "000178871722000003"},
    "UIS": {"cik": "746838", "accession": "000074683823000045"},
    "OC": {"cik": "1370946", "accession": "000137094624000044"},
    "CNM": {"cik": "1856525", "accession": "000185652524000035"},
    "AAP": {"cik": "1158449", "accession": "000115844925000147"},
    "NVS": {"cik": "1114448", "accession": "000137036823000006"},
    "HURN": {"cik": "1289848", "accession": "000128984824000092"},
    # Clean group
    "CTAS": {"cik": "723254", "accession": ""},
    "SHW": {"cik": "89800", "accession": ""},
    "ROK": {"cik": "1024478", "accession": ""},
}

# Case + backup companies to process
TARGET_COMPANIES = ["ALV", "MYE", "FXLV", "UIS", "OC", "NVS"]

MIN_IMAGE_SIZE = 5000  # bytes - skip tiny icons/logos


def extract_images_from_filing(ticker: str) -> list[dict]:
    """Extract and download chart images from a company's filing."""
    filing_dir = FILINGS_DIR / ticker / "filing"
    charts_out = CHARTS_DIR / ticker
    charts_out.mkdir(parents=True, exist_ok=True)

    meta = COMPANY_META.get(ticker)
    if not meta or not meta["accession"]:
        print(f"  [SKIP] {ticker}: no accession number")
        return []

    cik = meta["cik"]
    accession = meta["accession"]
    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"

    images = []

    for htm_file in filing_dir.glob("*.htm"):
        html = htm_file.read_text(encoding="utf-8", errors="replace")
        srcs = re.findall(r'src=["\']([^"\'> ]+)["\']', html, re.IGNORECASE)

        for src in srcs:
            if src.startswith("http"):
                url = src
            elif src.startswith("data:"):
                continue
            else:
                url = f"{base_url}/{src}"

            img_name = src.split("/")[-1].split("?")[0]
            local_path = charts_out / img_name

            if local_path.exists() and local_path.stat().st_size > MIN_IMAGE_SIZE:
                images.append({
                    "ticker": ticker,
                    "file": htm_file.name,
                    "image_name": img_name,
                    "local_path": str(local_path),
                })
                continue

            # Download
            try:
                with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        local_path.write_bytes(resp.content)
                        size = len(resp.content)
                        if size > MIN_IMAGE_SIZE:
                            images.append({
                                "ticker": ticker,
                                "file": htm_file.name,
                                "image_name": img_name,
                                "local_path": str(local_path),
                            })
                            print(f"    [OK] {img_name} ({size/1024:.0f} KB)")
                        else:
                            print(f"    [SKIP] {img_name} too small ({size} bytes)")
                    else:
                        print(f"    [FAIL] {img_name} HTTP {resp.status_code}")
            except Exception as e:
                print(f"    [ERROR] {img_name}: {e}")

            time.sleep(0.3)

    return images


def run_t2_on_chart(image_path: str) -> dict:
    """Run T2 Pipeline on a single chart via subprocess."""
    worker = Path("_pipeline_worker.py")
    if not worker.exists():
        # Write worker if not present
        from run_pipeline_ablation import write_worker_script
        write_worker_script()

    result = subprocess.run(
        [PYTHON, "-X", "utf8", str(worker), image_path],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
    )

    if result.returncode != 0:
        return {"error": result.stderr[-300:] if result.stderr else "unknown"}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"JSON parse: {result.stdout[-200:]}"}


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing T3 results
    t3_case = Path("data/eval_results/t3_case_study/results.json")
    t3_results = json.loads(t3_case.read_text(encoding="utf-8")) if t3_case.exists() else {}

    all_results = {}

    for ticker in TARGET_COMPANIES:
        print(f"\n{'='*60}")
        print(f"[{ticker}] Extracting charts from filing")
        print(f"{'='*60}")

        # Step 1: Extract images
        images = extract_images_from_filing(ticker)
        print(f"  Found {len(images)} chart-sized images")

        if not images:
            all_results[ticker] = {
                "ticker": ticker,
                "charts_found": 0,
                "t2_findings": [],
                "t3_findings": t3_results.get(f"{ticker}/{list(FILINGS_DIR.glob(f'{ticker}/filing/*.htm'))[0].name}" if list(FILINGS_DIR.glob(f"{ticker}/filing/*.htm")) else "", {}),
            }
            continue

        # Step 2: Run T2 Pipeline on each chart
        t2_findings = []
        for img in images:
            print(f"  T2 analyzing: {img['image_name']}", end="", flush=True)
            try:
                result = run_t2_on_chart(img["local_path"])
                if "error" in result:
                    print(f" -> ERROR: {result['error'][:60]}")
                    t2_findings.append({**img, "error": result["error"]})
                else:
                    predicted = result.get("predicted", [])
                    print(f" -> {predicted} ({result.get('elapsed_s', '?')}s)")
                    t2_findings.append({
                        **img,
                        "predicted": predicted,
                        "findings_count": result.get("findings_count", 0),
                        "elapsed_s": result.get("elapsed_s", 0),
                    })
            except subprocess.TimeoutExpired:
                print(f" -> TIMEOUT")
                t2_findings.append({**img, "error": "timeout"})

        # Step 3: Combine with T3
        # Find T3 result for this company
        t3_key = None
        for key in t3_results:
            if key.startswith(f"{ticker}/"):
                t3_key = key
                break

        t3_data = t3_results.get(t3_key, {}) if t3_key else {}

        all_results[ticker] = {
            "ticker": ticker,
            "charts_found": len(images),
            "charts_analyzed": len([f for f in t2_findings if "error" not in f]),
            "t2_findings": t2_findings,
            "t3_findings_count": t3_data.get("findings_count", 0),
            "t3_pairings": t3_data.get("pairing_count", 0),
            "t3_findings": t3_data.get("findings", []),
        }

        # Print company summary
        t2_issues = sum(1 for f in t2_findings if f.get("findings_count", 0) > 0)
        print(f"\n  Summary for {ticker}:")
        print(f"    Charts: {len(images)}, T2 issues: {t2_issues}")
        print(f"    T3 findings: {t3_data.get('findings_count', 'N/A')}")

    # Save full results
    output = EVAL_DIR / "full_audit_results.json"
    output.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"FULL AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Ticker':<8} {'Charts':>7} {'T2 Issues':>10} {'T3 Findings':>12}")
    print("-" * 40)
    for ticker, data in all_results.items():
        t2_issues = sum(1 for f in data["t2_findings"] if f.get("findings_count", 0) > 0)
        print(f"{ticker:<8} {data['charts_found']:>7} {t2_issues:>10} {data['t3_findings_count']:>12}")

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
