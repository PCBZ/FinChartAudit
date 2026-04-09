# data/download_sec_data.py

import requests
import json
from pathlib import Path


class SECDownloader:
    """Download SEC filings and comment letters."""

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

    def __init__(
        self,
        output_dir: str = 'data/sec',
        submissions_url: str = 'https://data.sec.gov/submissions/CIK{cik}.json',
        archives_url: str = 'https://www.sec.gov/Archives/edgar/data',
        user_agent: str = 'FinChartAudit your_email@northeastern.edu',
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_url = submissions_url
        self.archives_url = archives_url
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
        }

    def get_filings(self, cik: str, filing_type: str = '10-K', count: int = 10) -> list[dict]:
        """Get list of SEC filings using the EDGAR submissions REST API."""
        cik_padded = cik.zfill(10)
        url = self.submissions_url.format(cik=cik_padded)

        try:
            resp = requests.get(url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()

            recent     = data.get('filings', {}).get('recent', {})
            forms      = recent.get('form', [])
            dates      = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])
            primary_docs = recent.get('primaryDocument', [])


            results = []
            for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
                if form == filing_type:
                    results.append({
                        'form': form,
                        'filingDate': date,
                        'accessionNumber': acc,
                        'primaryDocument': doc,
                    })
                if len(results) >= count:
                    break
            return results

        except Exception as e:
            print(f"✗ Error fetching filings: {e}")
            return []

    def get_comments(self, cik: str, count: int = 10,
                     start: str = '2023-01-01', end: str = '2024-12-31') -> list[dict]:
        """Get comment letters via submissions API."""
        cik_padded = cik.zfill(10)
        url = self.submissions_url.format(cik=cik_padded)

        try:
            resp = requests.get(url, headers=self.headers)
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
    
    def download_pdf(self, ticker: str, accession: str, primary_doc: str) -> bytes:
        """Download the PDF content of a specific filing or comment letter."""
        cik = self.COMPANIES[ticker].lstrip('0')
        acc_no_dashes = accession.replace('-', '')
        url = f"{self.archives_url}/{cik}/{acc_no_dashes}/{primary_doc}"
        try:
            resp = requests.get(url, headers=self.headers)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"✗ Error downloading PDF: {e}")
            return b""


def download_sec_filings(submissions_url: str = None, archives_url: str = None, user_agent: str = None):
    """Download SEC filings and PDFs.
    
    Args:
        submissions_url: URL template for SEC submissions API (with {cik} placeholder)
        archives_url: Base URL for SEC Archives
        user_agent: User agent string for HTTP requests
    """
    # Use module defaults if not provided
    if submissions_url is None:
        submissions_url = 'https://data.sec.gov/submissions/CIK{cik}.json'
    if archives_url is None:
        archives_url = 'https://www.sec.gov/Archives/edgar/data'
    if user_agent is None:
        user_agent = 'FinChartAudit your_email@northeastern.edu'
    
    downloader = SECDownloader(
        submissions_url=submissions_url,
        archives_url=archives_url,
        user_agent=user_agent,
    )
    for ticker in SECDownloader.COMPANIES:
        downloader.download_company_data(ticker, max_10k=3, max_comments=10)
    
    for ticker in SECDownloader.COMPANIES:
        meta_path = Path('data/sec') / f"{ticker}.json"
        meta = json.loads(meta_path.read_text())
        
        out_dir = Path('data/pdfs') / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        for filing in meta.get('filings_10k', []):
            acc = filing['accessionNumber']
            doc = filing['primaryDocument']
            date = filing['filingDate']
            
            out_path = out_dir / f"{date}_{doc}"
            if out_path.exists():
                print(f"  ✓ Already exists: {out_path.name}")
                continue

            content = downloader.download_pdf(ticker, acc, doc)
            if content:
                out_path.write_bytes(content)
                print(f"  ✓ {ticker} {date} → {out_path.name} ({len(content)//1024} KB)")


def main(submissions_url: str = None, archives_url: str = None, user_agent: str = None):
    """Backward compatible wrapper for download_sec_filings()."""
    download_sec_filings(submissions_url, archives_url, user_agent)


if __name__ == "__main__":
    main()