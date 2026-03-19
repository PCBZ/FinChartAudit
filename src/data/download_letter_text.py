# data/download_letter_text.py

import requests
import json
import time
import re
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'FinChartAudit your_email@northeastern.edu',
    'Accept-Encoding': 'gzip, deflate',
}

SEC_BASE = "https://www.sec.gov/Archives/edgar/data"


def acc_to_path(accession: str) -> str:
    return accession.replace('-', '')


def get_primary_doc(cik_int: str, accession: str) -> tuple[str, bytes] | None:
    """
    Fetch the primary document of any filing by parsing the -index.htm page.
    Works for both UPLOAD and CORRESP.
    """
    acc_path = acc_to_path(accession)
    index_url = f"{SEC_BASE}/{cik_int}/{acc_path}/{accession}-index.htm"

    resp = requests.get(index_url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        # UPLOAD filings (accession 0000000000-xx-xx) use filename1.pdf directly
        for fname in ['filename1.pdf', 'filename1.txt', 'filename2.txt']:
            url = f"{SEC_BASE}/{cik_int}/{acc_path}/{fname}"
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                return fname, r.content
        return None

    # parse index page to find primary document filename
    soup = BeautifulSoup(resp.text, 'html.parser')
    for row in soup.select('table tr'):
        cells = row.find_all('td')
        if len(cells) >= 3:
            doc_type = cells[0].get_text(strip=True)
            link = cells[2].find('a')
            if link and doc_type in ('', 'CORRESP', 'UPLOAD', 'Letter'):
                fname = link['href'].split('/')[-1]
                file_url = f"{SEC_BASE}/{cik_int}/{acc_path}/{fname}"
                r = requests.get(file_url, headers=HEADERS, timeout=15)
                if r.status_code == 200:
                    return fname, r.content

    return None


def download_all_letters(sec_dir: str = 'data/sec',
                         letters_dir: str = 'data/letters'):
    sec_path = Path(sec_dir)
    out_base = Path(letters_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    total_ok, total_fail = 0, 0

    for json_path in sorted(sec_path.glob('*.json')):
        data = json.loads(json_path.read_text())
        ticker = data['ticker']
        cik = data['cik']
        cik_int = str(int(cik))
        comments = data['comment_letters']

        if not comments:
            continue

        ticker_dir = out_base / ticker
        ticker_dir.mkdir(exist_ok=True)

        print(f"\n{ticker} — {len(comments)} letter(s)")
        for c in comments:
            acc  = c['accessionNumber']
            form = c['form']
            date = c['filingDate']

            result = get_primary_doc(cik_int, acc)
            if result:
                fname, content = result
                suffix = Path(fname).suffix
                out_path = ticker_dir / f"{date}_{form}_{acc}{suffix}"
                out_path.write_bytes(content)
                size_kb = len(content) // 1024
                print(f"  ✓ {date} | {form:<8} | {out_path.name} ({size_kb}KB)")
                total_ok += 1
            else:
                print(f"  ✗ {date} | {form:<8} | {acc} — not found")
                total_fail += 1

            time.sleep(0.4)

    print(f"\n{'='*50}")
    print(f"Done: {total_ok} downloaded, {total_fail} failed")


if __name__ == "__main__":
    download_all_letters()