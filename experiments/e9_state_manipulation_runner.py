import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_URL = "https://api.typesafe.ai/v1/systemone"
API_KEY = os.getenv("TYPESAFE_API_KEY")
MODEL = "jev-latest"

INPUT_FILE = "experiments/data/e9_state_manipulation_dataset.csv"
OUTPUT_FILE = "experiments/data/e9_state_manipulation_results.csv"

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")

df = pd.read_csv(INPUT_FILE)

# Resume from existing results if the file already exists.
if os.path.exists(OUTPUT_FILE):
    results = pd.read_csv(OUTPUT_FILE)
    completed_ids = set(results["row_id"].astype(str))
    print(f"Resuming: {len(completed_ids)} rows already completed.")
else:
    results = pd.DataFrame()
    completed_ids = set()

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
})

for _, row in tqdm(df.iterrows(), total=len(df), desc="Running E9"):

    row_id = f"{row['case_id']}_{row['condition']}"

    if row_id in completed_ids:
        continue

    payload = {
        "model": MODEL,
        "state": row["state"],
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
        response = session.post(
            API_URL,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        probability = float(data["answers"]["q1"]["noul"])
        prediction = int(probability >= 0.5)
        label = int(row["label"])

        result = {
            "row_id": row_id,
            "case_id": row["case_id"],
            "domain": row["domain"],
            "condition": row["condition"],
            "label": label,
            "probability": probability,
            "prediction": prediction,
            "correct": int(prediction == label),
            "confidence": max(probability, 1 - probability),
            "model": data.get("model", MODEL),
            "input_tokens": data.get("usage", {}).get("input_tokens"),
            "output_tokens": data.get("usage", {}).get("output_tokens"),
        }

    except Exception as e:

        print(f"\nERROR on {row_id}: {e}")

        result = {
            "row_id": row_id,
            "case_id": row["case_id"],
            "domain": row["domain"],
            "condition": row["condition"],
            "label": int(row["label"]),
            "probability": None,
            "prediction": None,
            "correct": None,
            "confidence": None,
            "model": None,
            "input_tokens": None,
            "output_tokens": None,
            "error": str(e),
        }

    results = pd.concat(
        [results, pd.DataFrame([result])],
        ignore_index=True,
    )

    results.to_csv(OUTPUT_FILE, index=False)

    time.sleep(0.1)

print("\nE9 run complete.")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"Rows completed: {len(results)}")
print(f"Successful probability outputs: {results['probability'].notna().sum()}")
print(f"Errors: {results['probability'].isna().sum()}")
