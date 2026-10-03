import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")


DATASET_FILE = "experiments/data/e6_domain_shift_dataset.csv"
RESULTS_FILE = "experiments/data/e6_domain_shift_results.csv"

API_URL = "https://api.typesafe.ai/v1/systemone"


def run_case(row):

    payload = {
        "model": "jev-latest",
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
    probability = float(answer["noul"])

    prediction = int(probability >= 0.5)
    label = int(row["label"])
    correct = int(prediction == label)

    usage = data.get("usage", {})

    return {
        "probability": probability,
        "prediction": prediction,
        "correct": correct,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def main():

    df = pd.read_csv(DATASET_FILE)
    results = pd.read_csv(RESULTS_FILE)

    # Make sure the columns we will modify have compatible types.
    results["probability"] = pd.to_numeric(
        results["probability"],
        errors="coerce",
    )

    results["prediction"] = pd.to_numeric(
        results["prediction"],
        errors="coerce",
    )

    results["correct"] = pd.to_numeric(
        results["correct"],
        errors="coerce",
    )

    results["input_tokens"] = pd.to_numeric(
        results["input_tokens"],
        errors="coerce",
    )

    results["output_tokens"] = pd.to_numeric(
        results["output_tokens"],
        errors="coerce",
    )

    if "error" not in results.columns:
        results["error"] = pd.Series(
            index=results.index,
            dtype="object",
        )
    else:
        results["error"] = results["error"].astype("object")

    c27 = df[df["core_id"] == "C27"].copy()

    print("=" * 70)
    print("E6 C27 CORRECTION RERUN")
    print("=" * 70)

    for _, row in c27.iterrows():

        print()
        print(f"Running {row['case_id']}")
        print(f"State:    {row['state']}")
        print(f"Question: {row['question']}")
        print(f"Label:    {row['label']}")

        try:

            result = run_case(row)

            mask = results["case_id"] == row["case_id"]

            results.loc[mask, "probability"] = result["probability"]
            results.loc[mask, "prediction"] = result["prediction"]
            results.loc[mask, "correct"] = result["correct"]
            results.loc[mask, "input_tokens"] = result["input_tokens"]
            results.loc[mask, "output_tokens"] = result["output_tokens"]
            results.loc[mask, "error"] = None

            print(
                f"Probability: {result['probability']:.4f}"
            )
            print(
                f"Prediction:  {result['prediction']}"
            )
            print(
                f"Correct:     {result['correct']}"
            )

        except Exception as e:

            print(f"ERROR: {e}")

            mask = results["case_id"] == row["case_id"]

            results.loc[mask, "error"] = str(e)

        time.sleep(1)

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("C27 RERUN COMPLETE")
    print("=" * 70)
    print()
    print(f"Updated results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()