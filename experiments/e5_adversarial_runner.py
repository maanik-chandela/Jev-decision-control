import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# Frozen E5 dataset — READ ONLY
INPUT_FILE = "experiments/data/e5_adversarial_dataset.csv"

# Separate API results file
OUTPUT_FILE = "experiments/data/e5_adversarial_results.csv"

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")

df = pd.read_csv(INPUT_FILE)

# Verify the frozen dataset schema before making any API calls.
required_columns = [
    "counterfactual_id",
    "original_id",
    "domain",
    "perturbation_type",
    "state",
    "question",
    "label",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing required columns in E5 dataset: {missing_columns}"
    )

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

results = []

total_input_tokens = 0
total_output_tokens = 0

print("=" * 70)
print("E5 ADVERSARIAL ROBUSTNESS API RUN")
print("=" * 70)
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

        input_tokens = int(
            usage.get("input_tokens", 0)
        )

        output_tokens = int(
            usage.get("output_tokens", 0)
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        results.append({
            "counterfactual_id": row["counterfactual_id"],
            "original_id": row["original_id"],
            "domain": row["domain"],
            "perturbation_type": row["perturbation_type"],
            "label": int(row["label"]),
            "probability": probability,
            "prediction": prediction,
            "correct": int(
                prediction == int(row["label"])
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

        print(
            f"{i + 1:03d}/{len(df)} "
            f"{row['counterfactual_id']} "
            f"{row['perturbation_type']:<20} "
            f"p={probability:.2f} "
            f"pred={prediction} "
            f"label={int(row['label'])}"
        )

    except Exception as e:

        print(
            f"{i + 1:03d}/{len(df)} "
            f"{row['counterfactual_id']} "
            f"ERROR: {e}"
        )

        results.append({
            "counterfactual_id": row["counterfactual_id"],
            "original_id": row["original_id"],
            "domain": row["domain"],
            "perturbation_type": row["perturbation_type"],
            "label": int(row["label"]),
            "probability": None,
            "prediction": None,
            "correct": None,
            "input_tokens": 0,
            "output_tokens": 0,
        })

    time.sleep(0.2)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# COST
# ------------------------------------------------------------

estimated_cost = (
    total_input_tokens / 1_000_000_000
) * 42


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

successful = result_df["probability"].notna().sum()
failed = result_df["probability"].isna().sum()

print()
print("=" * 70)
print("E5 API RUN COMPLETE")
print("=" * 70)

print(f"Total evaluations:       {len(result_df)}")
print(f"Successful evaluations:  {successful}")
print(f"Failed evaluations:      {failed}")
print(f"Input tokens:            {total_input_tokens:,}")
print(f"Output tokens:           {total_output_tokens:,}")
print(f"Estimated cost:          ${estimated_cost:.8f}")
print(f"Output:                  {OUTPUT_FILE}")