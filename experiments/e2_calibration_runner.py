import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR / "experiments" / "data" / "e2_calibration_dataset.csv"
OUTPUT_PATH = BASE_DIR / "experiments" / "data" / "e2_calibration_results.csv"

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "TYPESAFE_API_KEY not found. Check your .env file."
    )


# ============================================================
# API CALL
# ============================================================

def call_jev(state: str, question: str) -> dict:
    """
    Send one decision question to JEV and return the raw answer
    plus token usage.
    """

    payload = {
        "state": state,
        "model": MODEL,
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": question,
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

    answer = data["answers"]["q1"]

    # JEV's noul answer contains the probability.
    probability = float(answer["noul"])

    # JEV decision:
    # >= 0.5 -> True
    # < 0.5  -> False
    prediction = int(probability >= 0.5)

    usage = data.get("usage", {})

    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))

    return {
        "probability": probability,
        "prediction": prediction,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ============================================================
# MAIN RUNNER
# ============================================================

def main():

    print("=" * 60)
    print("E2 CALIBRATION RUNNER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load frozen dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {
        "original_id",
        "domain",
        "difficulty",
        "state",
        "question",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    print(f"Dataset: {DATASET_PATH}")
    print(f"Cases: {len(df)}")

    # --------------------------------------------------------
    # Freeze checks
    # --------------------------------------------------------

    assert len(df) == 100, "Expected exactly 100 cases."
    assert df["label"].isin([0, 1]).all(), "Labels must be 0 or 1."
    assert df["label"].sum() == 50, "Expected 50 true labels."
    assert (df["label"] == 0).sum() == 50, "Expected 50 false labels."

    print("Dataset validation: PASSED")
    print("True labels: 50")
    print("False labels: 50")
    print()

    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    if OUTPUT_PATH.exists():
        raise RuntimeError(
            f"Results file already exists:\n{OUTPUT_PATH}\n\n"
            "Delete/rename it only if you intentionally want to "
            "rerun E2 from scratch."
        )

    # --------------------------------------------------------
    # Run JEV
    # --------------------------------------------------------

    results = []

    print("Starting 100 JEV calls...")
    print("The frozen dataset will NOT be modified.")
    print()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="E2 JEV",
    ):

        case_id = row["original_id"]
        domain = row["domain"]
        difficulty = row["difficulty"]
        state = row["state"]
        question = row["question"]
        label = int(row["label"])

        try:

            result = call_jev(
                state=state,
                question=question,
            )

            probability = result["probability"]
            prediction = result["prediction"]

            correct = int(prediction == label)

            results.append({
                "original_id": case_id,
                "domain": domain,
                "difficulty": difficulty,
                "label": label,
                "probability": probability,
                "prediction": prediction,
                "correct": correct,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "error": "",
            })

        except Exception as exc:

            print(f"\nERROR on {case_id}: {exc}")

            results.append({
                "case_id": case_id,
                "domain": domain,
                "difficulty": difficulty,
                "label": label,
                "probability": None,
                "prediction": None,
                "correct": None,
                "input_tokens": None,
                "output_tokens": None,
                "error": str(exc),
            })

        # Small delay to avoid unnecessarily aggressive requests.
        time.sleep(0.1)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    successful = results_df["probability"].notna().sum()
    failed = len(results_df) - successful

    print()
    print("=" * 60)
    print("E2 RUN COMPLETE")
    print("=" * 60)

    print(f"Total cases: {len(results_df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:

        valid = results_df.dropna(subset=["probability"])

        accuracy = valid["correct"].mean()

        total_input = valid["input_tokens"].sum()
        total_output = valid["output_tokens"].sum()

        # TypeSafe pricing:
        # $42 / 1 billion input tokens
        estimated_cost = total_input * 42 / 1_000_000_000

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Input tokens: {int(total_input):,}")
        print(f"Output tokens: {int(total_output):,}")
        print(f"Estimated input cost: ${estimated_cost:.8f}")

    print()
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()