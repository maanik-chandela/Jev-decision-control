import os
import time
import json
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

INPUT = Path("experiments/data/e4c_h_candidates.csv")
OUTPUT = Path("experiments/data/e4c_h_baseline_results.csv")
JSONL = Path("experiments/data/e4c_h_baseline_results.jsonl")

df = pd.read_csv(INPUT)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="E4c-H baseline"):
    payload = {
        "model": MODEL,
        "state": row["state"],
        "questions": {
            "answer": {
                "type": "noul",
                "instructions": row["question"],
                "criteria": {
                    "true": "The answer to the question is yes / true.",
                    "false": "The answer to the question is no / false.",
                },
            }
        },
    }

    response = requests.post(
        ENDPOINT,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    answer = data["answers"]["answer"]

    probability = float(answer["noul"])
    prediction = int(probability >= 0.5)
    label = int(row["label"])

    result = {
        "id": row["id"],
        "domain": row["domain"],
        "state": row["state"],
        "question": row["question"],
        "label": label,
        "probability": probability,
        "prediction": prediction,
        "correct": int(prediction == label),
        "uncertainty": 2 * min(probability, 1 - probability),
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
        "model": data.get("model", MODEL),
    }

    results.append(result)

    time.sleep(0.1)


out = pd.DataFrame(results)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)

with JSONL.open("w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

accuracy = out["correct"].mean()
total_input_tokens = out["input_tokens"].sum()
estimated_cost = total_input_tokens * 42 / 1_000_000_000

print()
print("=" * 60)
print("E4c-H BASELINE SCREENING")
print("=" * 60)

print(f"Evaluations: {len(out)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Input tokens: {total_input_tokens:,}")
print(f"Estimated cost: ${estimated_cost:.6f}")

print()
print("Probability distribution:")
print(out["probability"].describe())

print()
print("Cases ordered by uncertainty:")
print(
    out.sort_values("uncertainty", ascending=False)[
        ["id", "domain", "label", "probability", "prediction",
         "correct", "uncertainty"]
    ].head(25).to_string(index=False)
)

print()
print("Results written to:")
print(OUTPUT)
print(JSONL)

