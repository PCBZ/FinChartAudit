# data/download_sec_data.py

import requests
import json
from pathlib import Path
from typing import List, Dict
import time


class SECDownloader:
    """Download SEC filings and comment letters."""

    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    COMPANIES = {
        'TSLA': '0001318605',
        'LYFT': '0001759509',
        'ABNB': '0001559720',
        'RIVN': '0001874178',
        'VERI': '0001615165',
        'CHPT': '0001777393',
        'UPS':  '0001090727',
        'KMI':  '0001506307',
        'OGN':  '0001821825',
        'BCO':  '0000078890',
        'STZ':  '0000016918',
        'IRBT': '0001159167',
        'LITE': '0001633978',
    }

    def __init__(self, output_dir: str = 'data/sec'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'FinChartAudit your_email@northeastern.edu',
            'Accept-Encoding': 'gzip, deflate',
        }

    def get_filings(self, cik: str, filing_type: str = '10-K', count: int = 10) -> List[Dict]:
        """Get list of SEC filings using the EDGAR submissions REST API."""
        cik_padded = cik.zfill(10)
        url = self.SUBMISSIONS_URL.format(cik=cik_padded)

        try:
            time.sleep(0.5)
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            recent     = data.get('filings', {}).get('recent', {})
            forms      = recent.get('form', [])
            dates      = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])

            results = []
            for form, date, acc in zip(forms, dates, accessions):
                if form == filing_type:
                    results.append({
                        'form': form,
                        'filingDate': date,
                        'accessionNumber': acc,
                    })
                if len(results) >= count:
                    break
            return results

        except Exception as e:
            print(f"✗ Error fetching filings: {e}")
            return []

    def get_comments(self, cik: str, count: int = 10,
                     start: str = '2023-01-01', end: str = '2024-12-31') -> List[Dict]:
        """Get comment letters via submissions API."""
        cik_padded = cik.zfill(10)
        url = self.SUBMISSIONS_URL.format(cik=cik_padded)

        try:
            time.sleep(0.5)
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()

            recent     = resp.json().get('filings', {}).get('recent', {})
            forms      = recent.get('form', [])
            dates      = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])

            results = []
            for form, date, acc in zip(forms, dates, accessions):
                if form not in ('UPLOAD', 'CORRESP'):
                    continue
                if not (start <= date <= end):
                    continue
                results.append({'form': form, 'filingDate': date, 'accessionNumber': acc})
            return results[:count]

        except Exception as e:
            print(f"✗ Error fetching comments: {e}")
            return []

    def download_company_data(self, ticker: str, max_10k: int = 3, max_comments: int = 10):
        """Download all 10-K and comment letters for a company, save to JSON."""
        if ticker not in self.COMPANIES:
            print(f"✗ Unknown ticker: {ticker}")
            return False

        cik = self.COMPANIES[ticker]

        print(f"\n{'='*60}")
        print(f"Downloading SEC data for {ticker} (CIK: {cik})")
        print(f"{'='*60}\n")

        print("Fetching 10-K filings...")
        filings = self.get_filings(cik, '10-K', max_10k)
        if filings:
            print(f"✓ Found {len(filings)} 10-K filings")
            for i, f in enumerate(filings):
                print(f"  {i+1}. {f['filingDate']} - {f['accessionNumber']}")
        else:
            print("✗ No 10-K filings found")

        print("\nFetching comment letters...")
        comments = self.get_comments(cik, max_comments)
        if comments:
            print(f"✓ Found {len(comments)} comment letters")
            for i, c in enumerate(comments):
                print(f"  {i+1}. {c['filingDate']} | {c['form']:<8} | {c['accessionNumber']}")
        else:
            print("✗ No comment letters found (2023-2024)")

        # save to JSON
        out = {
            'ticker': ticker,
            'cik': cik,
            'filings_10k': filings,
            'comment_letters': comments,
        }
        out_path = self.output_dir / f"{ticker}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\n💾 Saved → {out_path}")

        print(f"\n{'='*60}\n")
        return True


def main():
    downloader = SECDownloader()
    for ticker in SECDownloader.COMPANIES:
        downloader.download_company_data(ticker, max_10k=3, max_comments=10)


if __name__ == "__main__":
    main()