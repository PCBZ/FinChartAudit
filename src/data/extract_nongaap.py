# data/extract_nongaap.py

import re
import json
from pathlib import Path


UPLOAD_KEYWORDS  = ['non-gaap', 'non gaap', 'nongaap']
CORRESP_KEYWORDS = ['non-gaap', 'non gaap', 'nongaap']
CONTEXT_WINDOW   = 2   # sentences before and after


def clean_text(raw: bytes) -> str:
    """Strip SGML headers and boilerplate from SEC filing text."""
    text = raw.decode('utf-8', errors='ignore')

    # remove SGML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # remove EDGAR header block (everything before first real paragraph)
    text = re.sub(r'(?s).*?(?=Dear\s|Re:\s|Ladies|Gentlemen|We\s(are|have|write)|This\s(letter|response))',
                  '', text, count=1, flags=re.IGNORECASE)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter — good enough for formal SEC letters."""
    # split on period/question mark followed by space + capital letter
    sentences = re.split(r'(?<=[.?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def extract_nongaap_mentions(sentences: list[str],
                              context: int = CONTEXT_WINDOW) -> list[dict]:
    """
    Find sentences containing Non-GAAP keywords,
    return each with surrounding context.
    """
    mentions = []
    seen_indices = set()

    for i, sent in enumerate(sentences):
        if any(kw in sent.lower() for kw in UPLOAD_KEYWORDS):
            # collect context window
            start = max(0, i - context)
            end   = min(len(sentences), i + context + 1)

            # avoid duplicate overlapping windows
            if i in seen_indices:
                continue
            seen_indices.update(range(start, end))

            mentions.append({
                'anchor_sentence': sent,
                'context': ' '.join(sentences[start:end]),
                'sentence_index': i,
            })

    return mentions


def process_letters(letters_dir: str = 'data/letters',
                    output_path: str = 'data/ground_truth.json'):
    base = Path(letters_dir)
    results = {}

    for ticker_dir in sorted(base.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        results[ticker] = []

        for txt_path in sorted(ticker_dir.glob('*.txt')):
            # parse filename: 2023-09-26_UPLOAD_0000000000-23-010608.txt
            parts = txt_path.stem.split('_')
            if len(parts) < 3:
                continue
            date, form, accession = parts[0], parts[1], '_'.join(parts[2:])

            raw = txt_path.read_bytes()
            text = clean_text(raw)
            sentences = split_sentences(text)
            mentions = extract_nongaap_mentions(sentences)

            if not mentions:
                continue

            results[ticker].append({
                'date':       date,
                'form':       form,
                'accession':  accession,
                'file':       txt_path.name,
                'total_nongaap_mentions': len(mentions),
                'mentions':   mentions,
            })

            print(f"  {ticker} | {date} | {form:<8} | {len(mentions)} mention(s)")

    # save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n💾 Saved → {out}")

    # summary
    total_files    = sum(len(v) for v in results.values())
    total_mentions = sum(e['total_nongaap_mentions']
                         for v in results.values() for e in v)
    print(f"📊 {total_files} files with Non-GAAP mentions, "
          f"{total_mentions} total mention windows")

    return results


def extract_all():
    """Extract Non-GAAP mentions from all comment letters."""
    return process_letters()


if __name__ == "__main__":
    extract_all()