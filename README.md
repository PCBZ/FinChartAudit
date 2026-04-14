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

## License

MIT
