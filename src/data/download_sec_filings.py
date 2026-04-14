"""
Download SEC filings and CORRESP documents based on SEC信息汇总.xlsx.

Downloads directly from the URLs provided in the spreadsheet.
Saves to data/filings/<TICKER>/ with organized subdirectories.
"""
import json
import time
import sys
from pathlib import Path

import httpx

# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
COMPANIES_FILE = DATA_DIR / "companies.json"

HEADERS = {
    "User-Agent": "FinChartAudit Research project@cornell.edu",
    "Accept-Encoding": "gzip, deflate",
}

# ---------- Company data from SEC信息汇总.xlsx ----------

COMPANIES = [
    # 备选组
    {
        "ticker": "NVS", "cik": "0001114448", "name": "Novartis AG",
        "sector": "Life Science", "group": "备选组", "year": "2022",
        "filing_type": "20-F",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1114448/000137036823000006/nvs-20221231.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/1114448/000110465923057754/filename1.htm",
        "issue": "非GAAP表格底部小字脚注标注",
    },
    {
        "ticker": "UDMY", "cik": "0001607939", "name": "Udemy, Inc.",
        "sector": "Technology", "group": "备选组", "year": "2023",
        "filing_type": "8-K",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1607939/000160793923000025/udmy-20230214.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/1607939/000160793923000077/filename1.htm",
        "issue": "完整非GAAP损益表，违反C&DI 102.10(c)",
    },
    {
        "ticker": "SOPH", "cik": "0001840706", "name": "SOPHiA GENETICS SA",
        "sector": "Life Science", "group": "备选组", "year": "2023",
        "filing_type": "6-K",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1840706/000184070623000014/sophiageneticssaex993q12023.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/1840706/000095010323007394/filename1.htm",
        "issue": "完整非IFRS损益表",
    },
    {
        "ticker": "HURN", "cik": "0001289848", "name": "Huron Consulting Group Inc.",
        "sector": "Professional Service", "group": "备选组", "year": "2023",
        "filing_type": "ARS",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1289848/000128984824000092/hurn_2023xars.pdf",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/1289848/000128984824000205/filename1.htm",
        "issue": "Adjusted EBITDA无GAAP对照",
    },
    # 案例组
    {
        "ticker": "ALV", "cik": "0001034670", "name": "Autoliv, Inc.",
        "sector": "Manufacturing", "group": "案例组", "year": "2022",
        "filing_type": "8-K",
        "filing_date": "2022-01-28",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1034670/000156459022002816/alv-8k_20220128.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/1034670/000119312522117804/filename1.htm",
        "corresp_date": "2022-04-25",
        "issue": "柱状图混合展示非GAAP指标未配套GAAP柱状图",
        "comment_location": "Comment 2",
    },
    {
        "ticker": "MYE", "cik": "0000069488", "name": "Myers Industries, Inc.",
        "sector": "Manufacturing", "group": "案例组", "year": "2022",
        "filing_type": "8-K",
        "filing_date": "2023-03-01",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/69488/000095017023005324/mye-ex99_1.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/69488/000095017023017823/filename1.htm",
        "corresp_date": "2023-05-04",
        "issue": "非GAAP利润率视觉突出，无GAAP对照",
        "comment_location": "Comment 1",
    },
    {
        "ticker": "FXLV", "cik": "0001788717", "name": "F45 Training Holdings Inc.",
        "sector": "Consumer Service", "group": "案例组", "year": "2021",
        "filing_type": "10-K",
        "filing_date": "2022-03-23",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1788717/000178871722000003/fxlv-20211231.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/1788717/000178871723000002/filename1.htm",
        "corresp_date": "2023-01-10",
        "issue": "Adjusted EBITDA margin无对应GAAP净利润率",
        "comment_location": "Comment 1",
    },
    {
        "ticker": "UIS", "cik": "0000746838", "name": "Unisys Corporation",
        "sector": "IT Service", "group": "案例组", "year": "2022",
        "filing_type": "DEF14A",
        "filing_date": "2023-03-24",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/746838/000074683823000045/filename1.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/746838/000074683823000045/filename1.htm",
        "corresp_date": "2023-09-21",
        "issue": "TSR对比图遗漏横轴年份标签",
        "comment_location": "Comment 1",
    },
    {
        "ticker": "OC", "cik": "0001370946", "name": "Owens Corning",
        "sector": "Manufacturing", "group": "案例组", "year": "2024",
        "filing_type": "8-K",
        "filing_date": "2024-02-14",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1370946/000137094624000044/a2023-12x31pressrelease.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/1370946/000119312524096323/filename1.htm",
        "corresp_date": "2024-04-15",
        "issue": "Adjusted EBIT/EBITDA Margin未展示GAAP Net Earnings Margin",
        "comment_location": "Comment 1",
    },
    {
        "ticker": "CNM", "cik": "0001856525", "name": "Core & Main, Inc.",
        "sector": "Industrial Distribution", "group": "案例组", "year": "2023",
        "filing_type": "8-K",
        "filing_date": "2024-03-19",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1856525/000185652524000035/cnm-20240319.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/1856525/000185652524000129/filename1.htm",
        "corresp_date": "2024-12-27",
        "issue": "投资者PPT仅展示非GAAP指标无GAAP图表",
        "comment_location": "Comment 2",
    },
    {
        "ticker": "AAP", "cik": "0001158449", "name": "Advance Auto Parts, Inc.",
        "sector": "Retail Auto Parts", "group": "案例组", "year": "2024",
        "filing_type": "8-K",
        "filing_date": "2025-02-26",
        "filing_url": "https://www.sec.gov/Archives/edgar/data/1158449/000115844925000147/filename1.htm",
        "corresp_url": "https://www.sec.gov/Archives/edgar/data/1158449/000115844925000147/filename1.htm",
        "corresp_date": "2025-04-10",
        "issue": "多个非GAAP YoY变化无对应GAAP数字",
        "comment_location": "Comment 3",
    },
    # 干净组
    {
        "ticker": "CTAS", "cik": "0000723254", "name": "Cintas Corporation",
        "sector": "Professional Service", "group": "干净组", "year": "",
        "filing_type": "", "filing_url": "", "corresp_url": "",
        "issue": "",
    },
    {
        "ticker": "SHW", "cik": "0000089800", "name": "Sherwin-Williams Company",
        "sector": "Consumer Service", "group": "干净组", "year": "",
        "filing_type": "", "filing_url": "", "corresp_url": "",
        "issue": "",
    },
    {
        "ticker": "ROK", "cik": "0001024478", "name": "Rockwell Automation",
        "sector": "Industrial Automation", "group": "干净组", "year": "",
        "filing_type": "", "filing_url": "", "corresp_url": "",
        "issue": "",
    },
]


def update_companies_json():
    """Replace placeholder companies.json with real SEC case companies."""
    registry = {}
    for c in COMPANIES:
        registry[c["ticker"]] = {
            "ticker": c["ticker"],
            "cik": c["cik"],
            "name": c["name"],
            "sector": c["sector"],
            "comment_letter_years": [c["year"]] if c.get("year") else [],
            "filings_downloaded": [],
            "notes": f"[{c['group']}] {c.get('issue', '')}".strip(),
        }

    COMPANIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    COMPANIES_FILE.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated companies.json with {len(registry)} companies")
    return registry


def download_url(client: httpx.Client, url: str, output_path: Path) -> bool:
    """Download a single URL to output_path."""
    if output_path.exists():
        print(f"  [SKIP] Already exists: {output_path.name}")
        return True

    try:
        resp = client.get(url, follow_redirects=True)
        if resp.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"  [OK] {output_path.name} ({size_kb:.0f} KB)")
            return True
        else:
            print(f"  [FAIL] {url} -> HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [ERROR] {url} -> {e}")
        return False


def guess_extension(url: str, content_type: str = "") -> str:
    """Guess file extension from URL or content type."""
    url_lower = url.lower()
    if url_lower.endswith(".pdf"):
        return ".pdf"
    elif url_lower.endswith(".htm") or url_lower.endswith(".html"):
        return ".htm"
    elif url_lower.endswith(".txt"):
        return ".txt"
    elif "pdf" in content_type:
        return ".pdf"
    else:
        return ".htm"


def download_all():
    """Download all filings and CORRESP documents."""
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    with httpx.Client(headers=HEADERS, timeout=60) as client:
        for c in COMPANIES:
            ticker = c["ticker"]
            group = c["group"]

            print(f"\n{'='*60}")
            print(f"[{group}] {c['name']} ({ticker})")
            print(f"{'='*60}")

            company_dir = FILINGS_DIR / ticker

            # --- Download filing document ---
            if c.get("filing_url"):
                url = c["filing_url"]
                ext = guess_extension(url)
                filing_type = c.get("filing_type", "filing").replace(" ", "_")
                year = c.get("year", "unknown")
                filename = f"{year}_{filing_type}{ext}"
                output = company_dir / "filing" / filename

                if download_url(client, url, output):
                    # Save metadata
                    meta = {
                        "ticker": ticker,
                        "type": "filing",
                        "form_type": filing_type,
                        "year": year,
                        "url": url,
                        "filing_date": c.get("filing_date", ""),
                        "issue": c.get("issue", ""),
                        "comment_location": c.get("comment_location", ""),
                    }
                    meta_path = company_dir / "filing" / f"{year}_{filing_type}_meta.json"
                    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                    stats["downloaded"] += 1
                else:
                    stats["failed"] += 1

                time.sleep(0.3)

            # --- Download CORRESP document ---
            if c.get("corresp_url") and c["corresp_url"] != c.get("filing_url", ""):
                url = c["corresp_url"]
                ext = guess_extension(url)
                year = c.get("year", "unknown")
                corresp_date = c.get("corresp_date", "")
                filename = f"CORRESP_{corresp_date or year}{ext}"
                output = company_dir / "corresp" / filename

                if download_url(client, url, output):
                    meta = {
                        "ticker": ticker,
                        "type": "corresp",
                        "url": url,
                        "date": corresp_date,
                    }
                    meta_path = company_dir / "corresp" / f"CORRESP_{corresp_date or year}_meta.json"
                    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                    stats["downloaded"] += 1
                else:
                    stats["failed"] += 1

                time.sleep(0.3)

            # --- For 干净组: download recent 10-K and 8-K for false positive testing ---
            if group == "干净组":
                print(f"  Searching EDGAR for recent filings...")
                cik_padded = c["cik"].lstrip("0").zfill(10)
                submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

                try:
                    resp = client.get(submissions_url)
                    resp.raise_for_status()
                    data = resp.json()
                    recent = data.get("filings", {}).get("recent", {})
                    forms = recent.get("form", [])
                    dates = recent.get("filingDate", [])
                    accessions = recent.get("accessionNumber", [])
                    primary_docs = recent.get("primaryDocument", [])

                    # Find most recent 10-K and 8-K
                    for target_form in ["10-K", "8-K"]:
                        for i, form in enumerate(forms):
                            if form == target_form:
                                acc_clean = accessions[i].replace("-", "")
                                cik_clean = c["cik"].lstrip("0")
                                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_clean}/{primary_docs[i]}"
                                ext = guess_extension(primary_docs[i])
                                filename = f"{dates[i]}_{target_form}{ext}"
                                output = company_dir / "filing" / filename

                                if download_url(client, doc_url, output):
                                    meta = {
                                        "ticker": ticker,
                                        "type": "filing",
                                        "form_type": target_form,
                                        "date": dates[i],
                                        "accession": accessions[i],
                                        "url": doc_url,
                                    }
                                    meta_path = company_dir / "filing" / f"{dates[i]}_{target_form}_meta.json"
                                    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                                    stats["downloaded"] += 1
                                else:
                                    stats["failed"] += 1

                                time.sleep(0.3)
                                break  # Only get most recent

                except Exception as e:
                    print(f"  [ERROR] EDGAR search failed: {e}")
                    stats["failed"] += 1

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Failed: {stats['failed']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Step 1: Updating companies.json...")
    update_companies_json()

    print("\nStep 2: Downloading SEC filings and CORRESP documents...")
    download_all()
