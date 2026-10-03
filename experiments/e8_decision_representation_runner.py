import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_URL = "https://api.typesafe.ai/v1/systemone"
API_KEY = os.getenv("TYPESAFE_API_KEY")

INPUT = Path("experiments/data/e8_decision_representation_dataset.csv")
OUTPUT = Path("experiments/data/e8_decision_representation_results.csv")

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")

df = pd.read_csv(INPUT)

# Resume safely if the experiment was interrupted.
if OUTPUT.exists():
    results = pd.read_csv(OUTPUT)
    completed = set(results["row_id"].astype(int))
else:
    results = pd.DataFrame()
    completed = set()

df["row_id"] = range(len(df))

pending = df[~df["row_id"].isin(completed)].copy()

print(f"Total evaluations: {len(df)}")
print(f"Already completed: {len(completed)}")
print(f"Remaining: {len(pending)}")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

for _, row in tqdm(
    pending.iterrows(),
    total=len(pending),
    desc="Running E8"
):
    payload = {
        "model": "jev-latest",
        "state": row["question"],
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": row["question"],
                "criteria": {
                    "true": "The statement is true.",
                    "false": "The statement is false.",
                },
            }
        },
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        probability = float(data["answers"]["q1"]["noul"])
        prediction = int(probability >= 0.5)
        correct = int(prediction == int(row["label"]))

        result = {
            "row_id": int(row["row_id"]),
            "case_id": row["case_id"],
            "domain": row["domain"],
            "representation": row["representation"],
            "label": int(row["label"]),
            "probability": probability,
            "prediction": prediction,
            "correct": correct,
            "model": data.get("model", "unknown"),
            "input_tokens": data.get("usage", {}).get("input_tokens"),
            "output_tokens": data.get("usage", {}).get("output_tokens"),
        }

    except Exception as e:
        result = {
            "row_id": int(row["row_id"]),
            "case_id": row["case_id"],
            "domain": row["domain"],
            "representation": row["representation"],
            "label": int(row["label"]),
            "probability": None,
            "prediction": None,
            "correct": None,
            "model": None,
            "input_tokens": None,
            "output_tokens": None,
            "error": str(e),
        }

    results = pd.concat(
        [results, pd.DataFrame([result])],
        ignore_index=True,
    )

    results.to_csv(OUTPUT, index=False)

    time.sleep(0.1)

print()
print(f"Saved results to: {OUTPUT}")
print(f"Total result rows: {len(results)}")

successful = results["probability"].notna().sum()
correct = results["correct"].sum()

print(f"Successful calls: {successful}/{len(df)}")
print(f"Correct: {int(correct)}/{successful}")
