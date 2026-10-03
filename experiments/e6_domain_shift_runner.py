import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

INPUT_FILE = "experiments/data/e6_domain_shift_dataset.csv"
OUTPUT_FILE = "experiments/data/e6_domain_shift_results.csv"

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in environment.")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "case_id",
    "core_id",
    "shift_group",
    "source_domain",
    "target_domain",
    "domain",
    "state",
    "question",
    "label",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

results = []

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

print("=" * 70)
print("E6 DOMAIN SHIFT EXPERIMENT")
print("=" * 70)
print(f"Cases: {len(df)}")
print(f"Endpoint: {ENDPOINT}")
print(f"Model: {MODEL}")
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
                    "false": "The statement is false.",
                },
            }
        },
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
        correct = int(prediction == int(row["label"]))

        usage = data.get("usage", {})

        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))

        results.append({
            "case_id": row["case_id"],
            "core_id": row["core_id"],
            "shift_group": row["shift_group"],
            "source_domain": row["source_domain"],
            "target_domain": row["target_domain"],
            "domain": row["domain"],
            "label": int(row["label"]),
            "probability": probability,
            "prediction": prediction,
            "correct": correct,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": "",
        })

        print(
            f"[{i + 1:03d}/{len(df)}] "
            f"{row['case_id']} | "
            f"{row['shift_group']:7s} | "
            f"p={probability:.3f} | "
            f"pred={prediction} | "
            f"correct={correct}"
        )

    except Exception as e:

        print(
            f"[{i + 1:03d}/{len(df)}] "
            f"{row['case_id']} | ERROR: {e}"
        )

        results.append({
            "case_id": row["case_id"],
            "core_id": row["core_id"],
            "shift_group": row["shift_group"],
            "source_domain": row["source_domain"],
            "target_domain": row["target_domain"],
            "domain": row["domain"],
            "label": int(row["label"]),
            "probability": None,
            "prediction": None,
            "correct": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(e),
        })

    time.sleep(0.2)


results_df = pd.DataFrame(results)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

successful = results_df["error"].eq("").sum()
failed = len(results_df) - successful

input_tokens = results_df["input_tokens"].sum()
output_tokens = results_df["output_tokens"].sum()

estimated_cost = input_tokens / 1_000_000_000 * 42

print()
print("=" * 70)
print("E6 RUN COMPLETE")
print("=" * 70)
print(f"Successful: {successful}/{len(df)}")
print(f"Failed:     {failed}/{len(df)}")
print(f"Input tokens:  {input_tokens:,}")
print(f"Output tokens: {output_tokens:,}")
print(f"Estimated input cost: ${estimated_cost:.8f}")
print()
print(f"Results written to:")
print(OUTPUT_FILE)
