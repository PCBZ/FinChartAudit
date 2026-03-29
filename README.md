# FinChartAudit

Evaluation framework for vision-language models on chart understanding in SEC 10-K filings.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
playwright install
```

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
# Then edit .env and replace placeholder values with your actual keys
```

## Usage

```bash
# Full pipeline
python src/run_pipeline.py

# Individual steps
python src/data/extract_charts.py
python src/data/extract_tables.py
python src/sec_pipeline.py          # Evaluate models
python src/visualization.py          # Generate visualizations
```