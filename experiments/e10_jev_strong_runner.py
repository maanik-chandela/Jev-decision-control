import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================
# E10 JEV STRONG ANSWER RUNNER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

DATASET_FILE = (
    PROJECT_ROOT
    / "experiments/data/e10_computation_allocation_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments/data/e10_jev_strong_results.csv"
)

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        f"TYPESAFE_API_KEY not found. Checked: {ENV_FILE}"
    )


# ============================================================
# API CALL
# ============================================================

def call_jev(question: str):
    """
    Ask JEV directly to determine whether the original
    proposition is TRUE or FALSE.

    This call is independent of the JEV controller call.
    """

    payload = {
        "model": MODEL,
        "state": question,
        "questions": {
            "final_decision": {
                "type": "noul",
                "instructions": (
                    "Determine whether the proposition in the state "
                    "is true or false. Return TRUE if the proposition "
                    "is true and FALSE if the proposition is false."
                ),
                "criteria": {
                    "true": "The proposition is true.",
                    "false": "The proposition is false.",
                },
            }
        },
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    answer = data["answers"]["final_decision"]

    probability_true = float(answer["noul"])

    prediction = 1 if probability_true >= 0.5 else 0

    usage = data.get("usage", {})

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    model_used = data.get("model")

    return {
        "jev_probability_true": probability_true,
        "jev_prediction": prediction,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "jev_model": model_used,
    }


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_FILE)

required_columns = {
    "case_id",
    "domain",
    "difficulty",
    "question",
    "label",
}

missing = required_columns - set(df.columns)

if missing:
    raise RuntimeError(
        f"Dataset missing required columns: {sorted(missing)}"
    )

df["case_id"] = df["case_id"].astype(str)

expected_cases = set(df["case_id"])

print("=" * 60)
print("E10 JEV STRONG ANSWER RUNNER")
print("=" * 60)

print(f"Dataset cases: {len(df)}")
print(f"Output file: {OUTPUT_FILE}")


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

if OUTPUT_FILE.exists():

    results = pd.read_csv(OUTPUT_FILE)

    if "case_id" in results.columns:
        results["case_id"] = results["case_id"].astype(str)

    # --------------------------------------------------------
    # IMPORTANT:
    # Only successful API calls count as completed.
    #
    # A row containing an API error must be retried.
    # --------------------------------------------------------

    successful_results = results[
        results["error"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    ]

    completed_cases = set(
        successful_results["case_id"]
    )

    # Keep only IDs that actually exist in the current dataset.
    completed_cases &= expected_cases

    print(
        f"Existing successfully completed cases: "
        f"{len(completed_cases)}"
    )

else:

    results = pd.DataFrame()

    completed_cases = set()

    print("No existing results found.")


# ============================================================
# DETERMINE REMAINING CASES
# ============================================================

remaining = df[
    ~df["case_id"].isin(completed_cases)
].copy()

print(f"Remaining JEV calls: {len(remaining)}")


# ============================================================
# RUN JEV
# ============================================================

new_rows = []

for _, row in tqdm(
    remaining.iterrows(),
    total=len(remaining),
    desc="JEV strong answers",
):

    case_id = row["case_id"]

    start_time = time.time()

    try:

        output = call_jev(
            row["question"]
        )

        elapsed = time.time() - start_time

        prediction = output["jev_prediction"]

        label = int(row["label"])

        correct = int(
            prediction == label
        )

        result = {
            "case_id": case_id,
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "question": row["question"],
            "label": label,

            "jev_probability_true": output[
                "jev_probability_true"
            ],

            "jev_prediction": prediction,

            "jev_correct": correct,

            "latency_seconds": elapsed,

            "input_tokens": output[
                "input_tokens"
            ],

            "output_tokens": output[
                "output_tokens"
            ],

            "jev_model": output[
                "jev_model"
            ],

            "error": "",
        }

    except Exception as e:

        elapsed = time.time() - start_time

        result = {
            "case_id": case_id,
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "question": row["question"],
            "label": int(row["label"]),

            "jev_probability_true": None,

            "jev_prediction": None,

            "jev_correct": None,

            "latency_seconds": elapsed,

            "input_tokens": None,

            "output_tokens": None,

            "jev_model": None,

            "error": str(e),
        }

    # --------------------------------------------------------
    # Add current result
    # --------------------------------------------------------

    new_rows.append(result)

    # --------------------------------------------------------
    # Save after EVERY API call.
    #
    # This means a crash or network failure will not lose
    # previously completed calls.
    # --------------------------------------------------------

    if results.empty:

        results = pd.DataFrame(
            [result]
        )

    else:

        results = pd.concat(
            [
                results,
                pd.DataFrame([result]),
            ],
            ignore_index=True,
        )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

results = pd.read_csv(
    OUTPUT_FILE
)

results["case_id"] = (
    results["case_id"]
    .astype(str)
)


# ------------------------------------------------------------
# Successful rows
# ------------------------------------------------------------

successful = results[
    results["error"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
].copy()


# ------------------------------------------------------------
# Failed rows
# ------------------------------------------------------------

failed = results[
    ~results.index.isin(
        successful.index
    )
].copy()


# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

if len(successful) > 0:

    successful["jev_correct"] = pd.to_numeric(
        successful["jev_correct"],
        errors="coerce",
    )

    accuracy = (
        successful["jev_correct"]
        .mean()
    )

else:

    accuracy = float("nan")


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("E10 JEV STRONG ANSWER COMPLETE")
print("=" * 60)

print(
    f"Dataset cases:             {len(df)}"
)

print(
    f"JEV results saved:         {len(results)}"
)

print(
    f"Successful JEV calls:      {len(successful)}"
)

print(
    f"Failed JEV calls:          {len(failed)}"
)

if len(successful) > 0:

    print(
        f"JEV strong accuracy:       "
        f"{accuracy:.4f}"
    )

    input_tokens = pd.to_numeric(
        successful["input_tokens"],
        errors="coerce",
    ).sum()

    output_tokens = pd.to_numeric(
        successful["output_tokens"],
        errors="coerce",
    ).sum()

    print(
        f"Total input tokens:        "
        f"{input_tokens:.0f}"
    )

    print(
        f"Total output tokens:       "
        f"{output_tokens:.0f}"
    )


# ------------------------------------------------------------
# Display failures
# ------------------------------------------------------------

if len(failed) > 0:

    print()
    print("-" * 60)
    print("FAILED CASES")
    print("-" * 60)

    for _, failed_row in failed.iterrows():

        print(
            f"{failed_row['case_id']}: "
            f"{failed_row['error']}"
        )


print()
print(
    f"Saved to: {OUTPUT_FILE}"
)

print("=" * 60)
