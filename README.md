# FinChartAudit

[![CI](https://img.shields.io/github/actions/workflow/status/PCBZ/FinChartAudit/ci.yml?branch=main&label=CI)](https://github.com/PCBZ/FinChartAudit/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20API-000000?logo=openai&logoColor=white)](https://openrouter.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Playwright](https://img.shields.io/badge/Playwright-Browser%20Automation-45BA4B?logo=playwright&logoColor=white)](https://playwright.dev/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-0A7E07)](LICENSE)

Detecting misleading charts in SEC financial filings with Vision-Language Models.

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

## System Overview

FinChartAudit evaluates whether multimodal models can detect misleading financial visuals and SEC Non-GAAP prominence issues.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
	'fontFamily': 'Trebuchet MS, Verdana, sans-serif',
	'primaryColor': '#e8f0ff',
	'primaryTextColor': '#0f172a',
	'lineColor': '#475569',
	'tertiaryColor': '#f8fafc'
}}}%%
flowchart TD
	A[SEC Filings + Letter Context] --> B[Data Preparation]
	B --> C[Chart and Table Extraction]
	C --> D{Evaluation Track}
	D --> E[RQ1/RQ2: MisViz]
	D --> F[RQ3: SEC]
	E --> G[vision_only and vision_text Prompts]
	F --> G
	G --> H[VLM Inference via OpenRouter]
	H --> I[Metric Aggregation]
	I --> J[Results JSON + Figures]

	classDef source fill:#fde68a,stroke:#b45309,stroke-width:2px,color:#451a03;
	classDef prep fill:#bfdbfe,stroke:#1d4ed8,stroke-width:2px,color:#0c4a6e;
	classDef track fill:#e9d5ff,stroke:#7e22ce,stroke-width:2px,color:#3b0764;
	classDef model fill:#bbf7d0,stroke:#15803d,stroke-width:2px,color:#14532d;
	classDef output fill:#fecdd3,stroke:#be123c,stroke-width:2px,color:#4c0519;

	class A source;
	class B,C prep;
	class D,E,F,G,H,I track;
	class H model;
	class J output;
```

## Data Card

- **MisViz (RQ1/RQ2)**: misleading chart benchmark with image-level annotations.
  - Labels: misleading / non-misleading + misleader types.
  - Data refs: [data/misviz/misviz.json](data/misviz/misviz.json), [data/charts](data/charts)

- **SEC corpus (RQ3)**: filing visuals and tables paired with SEC comment-letter context for Non-GAAP/compliance checks.
  - Data refs: [data/sec](data/sec), [data/charts](data/charts), [data/tables](data/tables), [data/letters](data/letters), [data/ground_truth.json](data/ground_truth.json)

- **Current evaluation coverage**
  - MisViz: `271` charts
  - SEC: `96` items across `10` tickers in aggregate output
  - Metrics source: [results/aggregated_summary.json](results/aggregated_summary.json)

## Model and Prompting

### Models

- `claude` → `anthropic/claude-haiku-4.5`
- `qwen` → `qwen/qwen3-vl-8b-instruct`

Configured in [src/config.py](src/config.py), called via OpenRouter in [src/vlm/openrouter_client.py](src/vlm/openrouter_client.py).

### Conditions

- **vision_only**: model sees only the image.
- **vision_text**: model sees the image plus structured text context (ground-truth data for MisViz or SEC context for RQ3).

### Prompt design

- Uses an explicit misleader taxonomy block and few-shot examples.
- Enforces strict JSON-only responses for stable parsing.
- SEC prompts separate chart and table logic, with Non-GAAP prominence rules based on SEC guidance.

Prompt builders are in [src/prompts.py](src/prompts.py):

- `build_vision_only_prompt`
- `build_vision_text_prompt`
- `build_chart_prompt`
- `build_table_prompt`
- `build_rq3_prompt`

## Results Summary

Primary aggregated metrics are saved in [results/aggregated_summary.json](results/aggregated_summary.json).

### RQ1/RQ2 (MisViz, 271 charts)

| Model | Condition | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Claude | vision_only | 0.771 | 0.782 | 0.883 | **0.830** |
| Claude | vision_text | **0.779** | 0.832 | 0.813 | 0.822 |
| Qwen | vision_only | 0.672 | 0.773 | 0.678 | 0.723 |
| Qwen | vision_text | 0.664 | **0.845** | 0.573 | 0.683 |

### RQ3 (SEC, 96 items)

| Model | Condition | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Claude | vision_only | 0.375 | 1.000 | 0.375 | 0.545 |
| Claude | vision_text | **0.396** | 1.000 | **0.396** | **0.567** |
| Qwen | vision_only | 0.167 | 1.000 | 0.167 | 0.286 |
| Qwen | vision_text | 0.073 | 1.000 | 0.073 | 0.136 |

### Quick takeaways

- Claude variants are stronger overall than Qwen on both MisViz and SEC tasks.
- On MisViz, `vision_text` slightly improves Claude accuracy, while `vision_only` gives Claude the best F1.
- On SEC, all models have high precision but low recall, indicating conservative flagging behavior (few false positives, many misses).

## License

MIT
