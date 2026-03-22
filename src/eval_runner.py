# src/eval_runner.py

import json
import os
import time
from pathlib import Path
from openai import OpenAI
from prompts import (
    build_vision_only_prompt,
    build_vision_text_prompt,
    build_bbox_text,
    MISLEADER_TYPES,
)

from datasets import load_dataset
from io import BytesIO
import base64
from huggingface_hub import login

# Model configurations
MODELS = {
    "claude": "anthropic/claude-sonnet-4-5",
    "qwen":   "qwen/qwen-2.5-vl-7b-instruct",
}

CONDITIONS = ("vision_only", "vision_text")


def parse_response(content: str) -> dict:
    """Extract JSON from model response, handling markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"misleading": None, "misleader_types": [], "explanation": content, "parse_error": True}


def run_single(client: OpenAI, model: str, image_url: str, condition: str, bbox_text: str = "") -> dict:
    """Run inference on a single sample."""
    prompt = build_vision_only_prompt() if condition == "vision_only" else build_vision_text_prompt(bbox_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            max_tokens=512,
        )
        if not response.choices:
            return {"misleading": None, "misleader_types": [], "explanation": "Empty response", "api_error": True}
        raw = response.choices[0].message.content
        if raw is None:
            return {"misleading": None, "misleader_types": [], "explanation": "Null content", "api_error": True}
        return parse_response(raw)
    except Exception as e:
        return {"misleading": None, "misleader_types": [], "explanation": str(e), "api_error": True}


def evaluate(api_key: str, model_key: str, condition: str, n_samples: int = None) -> list[dict]:
    """
    Run evaluation on Misviz dataset.

    Args:
        api_key:    OpenRouter API key
        model_key:  'claude' or 'qwen'
        condition:  'vision_only' or 'vision_text'
        n_samples:  number of samples (None = full 2604)
    """
    assert model_key in MODELS, f"model_key must be one of {list(MODELS)}"
    assert condition in CONDITIONS, f"condition must be one of {CONDITIONS}"

    model = MODELS[model_key]
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    dataset = load_dataset("UKPLab/misviz", split="train")
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    results = []
    correct, total = 0, 0

    print(f"Model: {model} | Condition: {condition} | Samples: {len(dataset)}")

    for i, item in enumerate(dataset):
        bboxes   = item.get("bbox", [])
        bbox_text = build_bbox_text(bboxes)

        sample = {
            "id":            i,
            "gt_misleaders": item.get("misleader", []),
            "chart_type":    item.get("chart_type", []),
            "bbox":          bboxes,
        }

        buf = BytesIO()
        item["image"].save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        pred = run_single(
            client=client,
            model=model,
            image_url=f"data:image/png;base64,{img_b64}",
            condition=condition,
            bbox_text=bbox_text,
        )

        gt_labels      = set(sample["gt_misleaders"])
        pred_labels    = set(pred.get("misleader_types", []))
        gt_binary      = len(gt_labels) > 0
        pred_binary    = pred.get("misleading", False)
        binary_correct = (gt_binary == pred_binary)
        type_match     = (gt_labels == pred_labels)

        if binary_correct:
            correct += 1
        total += 1

        results.append({
            **sample,
            "pred_misleading":      pred_binary,
            "pred_misleader_types": list(pred_labels),
            "explanation":          pred.get("explanation", ""),
            "binary_correct":       binary_correct,
            "type_match":           type_match,
            "parse_error":          pred.get("parse_error", False),
            "api_error":            pred.get("api_error", False),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(dataset)}] binary acc so far: {correct/total:.3f}")

        time.sleep(1.0)

    # Save
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{model_key}_{condition}.json"
    with open(fname, "w") as f:
        json.dump({
            "model":           model,
            "condition":       condition,
            "n_samples":       total,
            "binary_accuracy": correct / total if total else 0,
            "results":         results,
        }, f, indent=2)

    print(f"\nBinary accuracy: {correct/total:.3f} ({correct}/{total})")
    print(f"Saved → {fname}")
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    evaluate(api_key=os.environ["OPENROUTER_API_KEY"], model_key="qwen", condition="vision_text", n_samples=3)