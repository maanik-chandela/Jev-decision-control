import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

INPUT_FILE = "experiments/data/e4c_h_counterfactuals.csv"
OUTPUT_FILE = "experiments/data/e4c_h_counterfactual_results.csv"

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")

df = pd.read_csv(INPUT_FILE)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

results = []

total_input_tokens = 0
total_output_tokens = 0

print("=" * 60)
print("E4c-H COUNTERFACTUAL API RUN")
print("=" * 60)
print(f"Counterfactuals: {len(df)}")
print()

for i, row in df.iterrows():

    payload = {
        "model": MODEL,
        "state": row["state"],
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": row["question"],
                "criteria": {
                    "true": "The statement is true.",
                    "false": "The statement is false."
                }
            }
        }
    }

    try:
        response = requests.post(
            ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        answer = data["answers"]["q1"]

        probability = float(answer["noul"])
        prediction = int(probability >= 0.5)

        usage = data.get("usage", {})

        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        results.append({
            "counterfactual_id": row["counterfactual_id"],
            "original_id": row["original_id"],
            "perturbation_type": row["perturbation_type"],
            "confidence_group": row["confidence_group"],
            "original_probability": row["original_probability"],
            "label": int(row["label"]),
            "probability": probability,
            "prediction": prediction,
            "correct": int(prediction == int(row["label"])),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

        print(
            f"{i+1:02d}/{len(df)} "
            f"{row['counterfactual_id']} "
            f"p={probability:.2f} "
            f"pred={prediction} "
            f"label={int(row['label'])}"
        )

    except Exception as e:
        print(
            f"{i+1:02d}/{len(df)} "
            f"{row['counterfactual_id']} ERROR: {e}"
        )

        results.append({
            "counterfactual_id": row["counterfactual_id"],
            "original_id": row["original_id"],
            "perturbation_type": row["perturbation_type"],
            "confidence_group": row["confidence_group"],
            "original_probability": row["original_probability"],
            "label": int(row["label"]),
            "probability": None,
            "prediction": None,
            "correct": None,
            "input_tokens": 0,
            "output_tokens": 0,
        })

    time.sleep(0.2)

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)

estimated_cost = total_input_tokens / 1_000_000_000 * 42

print()
print("=" * 60)
print("E4c-H RUN COMPLETE")
print("=" * 60)
print(f"Successful evaluations: {result_df['probability'].notna().sum()}")
print(f"Failed evaluations: {result_df['probability'].isna().sum()}")
print(f"Input tokens: {total_input_tokens:,}")
print(f"Output tokens: {total_output_tokens:,}")
print(f"Estimated cost: ${estimated_cost:.8f}")
print(f"Output: {OUTPUT_FILE}")
