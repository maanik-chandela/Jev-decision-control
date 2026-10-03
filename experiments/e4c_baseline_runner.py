import csv
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")
API_URL = "https://api.typesafe.ai/v1/systemone"

INPUT = Path("experiments/data/e4c_candidates.csv")
OUTPUT_CSV = Path("experiments/data/e4c_baseline_results.csv")
OUTPUT_JSONL = Path("experiments/data/e4c_baseline_results.jsonl")

MODEL = "jev-latest"

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")

rows = []

with INPUT.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

results = []

for row in tqdm(rows, desc="E4c baseline"):
    payload = {
        "model": MODEL,
        "state": row["state"],
        "questions": {
            "answer": {
                "type": "noul",
                "instructions": row["question"],
                "criteria": {
                    "true": "The statement in the question is true.",
                    "false": "The statement in the question is false.",
                },
            }
        },
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    probability = float(data["answers"]["answer"]["noul"])
    true_label = int(row["true_label"])

    prediction = int(probability >= 0.5)
    correct = int(prediction == true_label)

    usage = data.get("usage", {})

    result = {
        "candidate_id": row["candidate_id"],
        "domain": row["domain"],
        "state": row["state"],
        "question": row["question"],
        "true_label": true_label,
        "baseline_probability": probability,
        "baseline_prediction": prediction,
        "correct": correct,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }

    results.append(result)

    time.sleep(0.1)

with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

total_tokens = sum(r["input_tokens"] for r in results)
correct = sum(r["correct"] for r in results)

print()
print("=" * 70)
print("E4c BASELINE COMPLETE")
print("=" * 70)
print(f"Evaluations: {len(results)}")
print(f"Correct: {correct}/{len(results)}")
print(f"Accuracy: {correct / len(results):.4f}")
print(f"Input tokens: {total_tokens}")
print()
print(f"CSV: {OUTPUT_CSV.resolve()}")
print(f"JSONL: {OUTPUT_JSONL.resolve()}")
