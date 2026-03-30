# data/download_letter_text.py

import requests
import json
from pathlib import Path
from bs4 import BeautifulSoup


class LetterDownloader:
    """Download comment letters from SEC filings."""

    def __init__(
        self,
        sec_base_url: str = 'https://www.sec.gov/Archives/edgar/data',
        user_agent: str = 'FinChartAudit your_email@northeastern.edu',
    ):
        self.sec_base = sec_base_url
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
        }

    @staticmethod
    def acc_to_path(accession: str) -> str:
        return accession.replace('-', '')

    def get_primary_doc(self, cik_int: str, accession: str) -> tuple[str, bytes] | None:
        """
        Fetch the primary document of any filing by parsing the -index.htm page.
        Works for both UPLOAD and CORRESP.
        """
        acc_path = self.acc_to_path(accession)
        index_url = f"{self.sec_base}/{cik_int}/{acc_path}/{accession}-index.htm"

        resp = requests.get(index_url, headers=self.headers)
        if resp.status_code != 200:
            # UPLOAD filings (accession 0000000000-xx-xx) use filename1.pdf directly
            for fname in ['filename1.pdf', 'filename1.txt', 'filename2.txt']:
                url = f"{self.sec_base}/{cik_int}/{acc_path}/{fname}"
                r = requests.get(url, headers=self.headers)
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
                    file_url = f"{self.sec_base}/{cik_int}/{acc_path}/{fname}"
                    r = requests.get(file_url, headers=self.headers)
                    if r.status_code == 200:
                        return fname, r.content

        return None

    def download_all_letters(self, sec_dir: str = 'data/sec',
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

                result = self.get_primary_doc(cik_int, acc)
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

        print(f"\n{'='*50}")
        print(f"Done: {total_ok} downloaded, {total_fail} failed")


def download_all_letters(sec_dir: str = 'data/sec',
                         letters_dir: str = 'data/letters',
                         sec_base_url: str = None,
                         user_agent: str = None):
    """Download comment letters from SEC filings.
    
    Args:
        sec_dir: Directory containing SEC metadata
        letters_dir: Output directory for downloaded letters
        sec_base_url: Base URL for SEC Archives
        user_agent: User agent string for HTTP requests
    """
    if sec_base_url is None:
        sec_base_url = 'https://www.sec.gov/Archives/edgar/data'
    if user_agent is None:
        user_agent = 'FinChartAudit your_email@northeastern.edu'
    
    downloader = LetterDownloader(sec_base_url=sec_base_url, user_agent=user_agent)
    downloader.download_all_letters(sec_dir, letters_dir)


if __name__ == "__main__":
    download_all_letters()