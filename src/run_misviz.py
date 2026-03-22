# src/run_misviz.py
# Runs the full 2x2 experiment on Misviz: (claude, qwen) x (vision_only, vision_text)

import os
from pathlib import Path
from dotenv import load_dotenv
from eval_runner import evaluate, MODELS, CONDITIONS

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY   = os.environ["OPENROUTER_API_KEY"]
N_SAMPLES = 3  # None = full 2604, set e.g. 50 for a quick test run

if __name__ == "__main__":
    for model_key in MODELS:
        for condition in CONDITIONS:
            print(f"\n{'='*60}")
            print(f"Running: {model_key} | {condition}")
            print(f"{'='*60}")
            evaluate(api_key=API_KEY, model_key=model_key, condition=condition, n_samples=N_SAMPLES)