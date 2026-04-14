# FinChartAudit

Detecting misleading charts in SEC financial filings with Vision-Language Models.

**CS 6180 Generative AI Capstone - Northeastern University - 2026**

## Project Structure

```
FinChartAudit/
├── src/                    # Core framework
│   ├── data/               # Data extraction (charts, tables, SEC filings)
│   ├── vlm/                # VLM client (OpenRouter)
│   ├── api/                # FastAPI production server
│   ├── classifier/         # ViT chart misleader classifier
│   ├── run_pipeline.py     # Main evaluation pipeline
│   ├── sec_pipeline.py     # SEC-specific pipeline
│   └── ...
├── experiments/            # Pipeline experiment variants (v3-v8)
├── ui/demo-ui/             # Next.js interactive demo
├── data/                   # Ground truth, SEC filings, Misviz
├── results/                # Evaluation results and figures
├── docs/                   # Design docs, experiment reports
├── requirements.txt        # Core dependencies
├── requirements-api.txt    # API server dependencies
└── pyproject.toml          # Package config with optional deps
```

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
```

## Usage

### Core Pipeline

```bash
python src/run_pipeline.py           # Full evaluation pipeline
python src/sec_pipeline.py           # SEC-specific evaluation
python src/visualization.py          # Generate result figures
```

### Experiments

> **Note**: All experiment scripts require the extended DD_v1 package layout
> (`finchartaudit.*`, `data_tools.*`) which is not part of this repo. They are
> preserved here for reproducibility and reference. To run them, set `PYTHONPATH`
> to include the DD_v1 project root.

```
experiments/
├── v3_veto.py             # ViT classifier veto
├── v4_combo.py            # Combined approach
├── v5_deplot.py           # DePlot integration
├── v6_targeted.py         # Targeted re-ask
├── v7_sequential.py       # Sequential pipeline
├── v8_selfconsist.py      # Self-consistency voting
├── full_pipeline.py       # Full OCR + VLM pipeline
├── ablation.py            # T2 Pipeline ablation study
├── ocr_rules.py           # LLM + OCR + Rules approach
├── vlm_rules.py           # VLM + Rules approach
├── sec_chart_comparison.py # SEC chart evaluation
└── sonnet_comparison.py   # Sonnet model comparison
```

### Demo (API + UI)

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Start API server
python src/api/server.py

# In another terminal — start the UI
cd ui/demo-ui
npm install
npm run dev
```

Open http://localhost:3000 to use the interactive demo:
- **Pre-Filing Check** - Upload charts/PDFs or fetch from SEC EDGAR by ticker
- **Audit Report** - View detailed compliance findings with severity levels
- **Risk Dashboard** - Cross-company flag rate comparison
- **Detection Reference** - 12 visual misleader types + 4 Non-GAAP violation types

### ViT Classifier Training

```bash
pip install -e ".[experiments]"
python src/classifier/train.py
```

## Architecture

**Best pipeline**: VLM Call 1 -> Rule Dedup -> Misrep Re-ask -> ViT Classifier Veto

The system detects:
- **Visual misleaders** (6 SEC-relevant types): truncated axis, misrepresentation, 3D distortion, dual axis, pie chart misuse, inconsistent tick intervals
- **Non-GAAP violations**: prominence violations (Item 10(e)), missing reconciliation (Reg G), labeling issues

## License

MIT
