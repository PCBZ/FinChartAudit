import os
from pathlib import Path
from dotenv import load_dotenv
from eval_runner import evaluate_sec

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
api_key = os.environ["OPENROUTER_API_KEY"]

# 4 experiments: 2 models × 2 conditions
for model_key in ("claude", "qwen"):
    for condition in ("vision_text", "vision_only"):
        print(f"\n{'#'*60}")
        print(f"  {model_key.upper()} | {condition}")
        print(f"{'#'*60}")
        evaluate_sec(
            api_key=api_key,
            model_key=model_key,
            condition=condition,
            max_per_ticker=3,
        )