import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

INPUT_PRICE_PER_BILLION = 42.0
BUDGET_LIMIT = 4.0

DATASET_PATH = "experiments/data/e12_temporal_stability_dataset.csv"
RESULTS_PATH = "experiments/data/e12_temporal_stability_results.csv"


# ============================================================
# Validation
# ============================================================

if not API_KEY:
    raise RuntimeError(
        "TYPESAFE_API_KEY not found in .env"
    )


# ============================================================
# Helper functions
# ============================================================

def estimate_cost(input_tokens):
    return (input_tokens / 1_000_000_000) * INPUT_PRICE_PER_BILLION


def get_existing_cost():
    if not os.path.exists(RESULTS_PATH):
        return 0.0

    try:
        df = pd.read_csv(RESULTS_PATH)

        if "estimated_cost_usd" not in df.columns:
            return 0.0

        return pd.to_numeric(
            df["estimated_cost_usd"],
            errors="coerce"
        ).fillna(0).sum()

    except Exception:
        return 0.0


def call_jev(question):
    payload = {
        "model": MODEL,
        "state": {},
        "questions": {
            "decision": {
                "type": "noul",
                "instructions": question
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    start = time.perf_counter()

    response = requests.post(
        API_URL,
        json=payload,
        headers=headers,
        timeout=60
    )

    latency_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()

    data = response.json()

    return data, latency_ms


def extract_probability(data):
    """
    Extract the probability from the JEV response.

    Expected structure is based on the TypeSafe System One API:
    responses -> decision -> noul -> probability

    The fallback traversal makes the runner tolerant to
    minor response-shape differences.
    """

    # Common expected structure
   # Actual TypeSafe JEV response structure
    try:
        return float(
            data["answers"]["decision"]["noul"])
    except (KeyError, TypeError, ValueError):
        pass

    # Alternative structure
    try:
        return float(
            data["questions"]["decision"]["probability"]
        )
    except (KeyError, TypeError, ValueError):
        pass

    # Recursive search for a probability-like field
    def recursive_search(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in {
                    "probability",
                    "prob",
                    "noul_probability"
                }:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass

                result = recursive_search(value)

                if result is not None:
                    return result

        elif isinstance(obj, list):
            for item in obj:
                result = recursive_search(item)

                if result is not None:
                    return result

        return None

    probability = recursive_search(data)

    if probability is None:
        raise ValueError(
            "Could not find probability in JEV response:\n"
            + str(data)
        )

    return probability


def extract_model(data):
    if isinstance(data, dict):
        if "model" in data:
            return data["model"]

        if "metadata" in data and isinstance(data["metadata"], dict):
            if "model" in data["metadata"]:
                return data["metadata"]["model"]

    return MODEL


def extract_tokens(data):
    """
    Try several common metadata locations.
    """

    if not isinstance(data, dict):
        return None, None

    candidates = []

    candidates.append(data.get("usage"))

    if isinstance(data.get("metadata"), dict):
        candidates.append(data["metadata"].get("usage"))

    for usage in candidates:
        if not isinstance(usage, dict):
            continue

        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("inputTokens")
        )

        output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("outputTokens")
        )

        return input_tokens, output_tokens

    return None, None


def prediction_from_probability(probability):
    return int(probability >= 0.5)


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "case_id",
    "domain",
    "difficulty",
    "label",
    "t1_question",
    "t2_question",
    "t3_question",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        f"Dataset is missing columns: {missing}"
    )


# ============================================================
# Resume support
# ============================================================

if os.path.exists(RESULTS_PATH):
    results_df = pd.read_csv(RESULTS_PATH)

    completed = set(
        zip(
            results_df["case_id"].astype(str),
            results_df["timepoint"].astype(str)
        )
    )

    print(
        f"Existing results found: {len(results_df)} rows"
    )

else:
    results_df = pd.DataFrame()
    completed = set()

    print("No existing results found. Starting from zero.")


# ============================================================
# Cost guard
# ============================================================

spent = get_existing_cost()

print(f"Recorded E12 spend: ${spent:.6f}")
print(f"Software budget limit: ${BUDGET_LIMIT:.2f}")

if spent >= BUDGET_LIMIT:
    raise RuntimeError(
        "E12 software budget limit reached. "
        "No API calls will be made."
    )


# ============================================================
# Execution
# ============================================================

timepoint_columns = {
    "T1": "t1_question",
    "T2": "t2_question",
    "T3": "t3_question",
}

total_calls = len(df) * 3

print()
print(f"Cases: {len(df)}")
print(f"Timepoints per case: 3")
print(f"Maximum API calls: {total_calls}")
print()
print("Evaluation order:")
print("T1 -> T2 -> T3")
print()


for row_index, row in df.iterrows():

    case_id = str(row["case_id"])

    for timepoint, question_column in timepoint_columns.items():

        key = (case_id, timepoint)

        if key in completed:
            print(
                f"[SKIP] {case_id} {timepoint} already completed"
            )
            continue

        question = str(row[question_column])

        current_spend = get_existing_cost()

        if current_spend >= BUDGET_LIMIT:
            print()
            print("Budget limit reached.")
            print(
                f"Recorded spend: ${current_spend:.6f}"
            )
            print("Stopping safely.")
            raise SystemExit(0)

        print(
            f"[RUN] {case_id} | {timepoint} | "
            f"{row['domain']} | {row['difficulty']}"
        )

        print(f"Question: {question}")

        timestamp = datetime.now(timezone.utc).isoformat()

        try:

            data, latency_ms = call_jev(question)

            probability = extract_probability(data)

            prediction = prediction_from_probability(
                probability
            )

            label = int(row["label"])

            correct = int(
                prediction == label
            )

            model_returned = extract_model(data)

            input_tokens, output_tokens = extract_tokens(data)

            if input_tokens is not None:
                estimated_cost = estimate_cost(
                    input_tokens
                )
            else:
                estimated_cost = 0.0

            result = {
                "case_id": case_id,
                "domain": row["domain"],
                "difficulty": row["difficulty"],
                "label": label,
                "timepoint": timepoint,
                "question": question,
                "probability": probability,
                "prediction": prediction,
                "correct": correct,
                "timestamp_utc": timestamp,
                "latency_ms": round(latency_ms, 2),
                "api_model": model_returned,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": estimated_cost,
            }

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame([result])
                ],
                ignore_index=True
            )

            results_df.to_csv(
                RESULTS_PATH,
                index=False
            )

            completed.add(key)

            print(
                f"  p={probability:.4f} "
                f"pred={prediction} "
                f"label={label} "
                f"correct={correct}"
            )

            print(
                f"  latency={latency_ms:.1f} ms"
            )

            if input_tokens is not None:
                print(
                    f"  input_tokens={input_tokens}"
                )

            print()

        except Exception as exc:

            print()
            print(
                f"[ERROR] {case_id} {timepoint}"
            )
            print(str(exc))
            print()

            # Save an error row so the failure is visible.
            error_result = {
                "case_id": case_id,
                "domain": row["domain"],
                "difficulty": row["difficulty"],
                "label": int(row["label"]),
                "timepoint": timepoint,
                "question": question,
                "probability": None,
                "prediction": None,
                "correct": None,
                "timestamp_utc": timestamp,
                "latency_ms": None,
                "api_model": None,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_usd": 0.0,
            }

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame([error_result])
                ],
                ignore_index=True
            )

            results_df.to_csv(
                RESULTS_PATH,
                index=False
            )

            # Do not continue silently after an API problem.
            raise


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 60)
print("E12 RUN COMPLETE")
print("=" * 60)

final_df = pd.read_csv(RESULTS_PATH)

successful = final_df[
    final_df["probability"].notna()
]

print(f"Rows recorded: {len(final_df)}")
print(f"Successful evaluations: {len(successful)}")
print(f"Expected evaluations: {len(df) * 3}")

print()
print(
    f"Recorded API input spend: "
    f"${successful['estimated_cost_usd'].sum():.6f}"
)

print()
print("Rows by timepoint:")

if len(successful):
    print(
        successful["timepoint"]
        .value_counts()
        .sort_index()
    )

print()
print(f"Results saved to:")
print(RESULTS_PATH)
