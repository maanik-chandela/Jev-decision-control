import os
import pandas as pd
import requests
from dotenv import load_dotenv

DATASET = "experiments/data/e7_language_shift_dataset.csv"
RESULTS = "experiments/data/e7_language_shift_results.csv"
API_URL = "https://api.typesafe.ai/v1/systemone"

TARGET_CASES = {"P09", "L03", "L05", "L09"}

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found")


def call_jev(question):
    payload = {
        "model": "jev-latest",
        "state": question,
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": question,
                "criteria": {
                    "true": "The statement is true.",
                    "false": "The statement is false.",
                },
            }
        },
    }

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    probability = float(data["answers"]["q1"]["noul"])

    return probability, data


def main():

    dataset = pd.read_csv(DATASET)
    results = pd.read_csv(RESULTS)

    target = dataset[
        dataset["core_id"].isin(TARGET_CASES)
    ].copy()

    print(f"Cases to rerun: {len(target)}")
    print(target["core_id"].value_counts())
    print(target["language"].value_counts())

    assert len(target) == 12
    assert len(results) == 180

    for _, row in target.iterrows():

        mask = results["case_id"] == row["case_id"]

        if mask.sum() != 1:
            raise RuntimeError(
                f"Could not uniquely locate {row['case_id']}"
            )

        idx = results.index[mask][0]

        probability, raw = call_jev(row["question"])

        prediction = int(probability >= 0.5)
        correct = int(prediction == int(row["label"]))

        results.at[idx, "label"] = row["label"]
        results.at[idx, "question"] = row["question"]
        results.at[idx, "probability"] = probability
        results.at[idx, "prediction"] = prediction
        results.at[idx, "correct"] = correct

        usage = raw.get("usage", {})

        if "input_tokens" in results.columns:
            results.at[idx, "input_tokens"] = usage.get(
                "input_tokens"
            )

        if "output_tokens" in results.columns:
            results.at[idx, "output_tokens"] = usage.get(
                "output_tokens"
            )

        if "model" in results.columns:
            results.at[idx, "model"] = raw.get("model")

        print(
            f"{row['case_id']} | "
            f"{row['language']} | "
            f"p={probability:.2f} | "
            f"label={int(row['label'])} | "
            f"correct={correct}"
        )

        # Save after every API call
        results.to_csv(RESULTS, index=False)

    print("\nCorrection run complete.")
    print("Updated 12 evaluations.")


if __name__ == "__main__":
    main()
