import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv


INPUT = "experiments/data/e7_language_shift_dataset.csv"
OUTPUT = "experiments/data/e7_language_shift_results.csv"

API_URL = "https://api.typesafe.ai/v1/systemone"

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")


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
    df = pd.read_csv(INPUT)

    if os.path.exists(OUTPUT):
        results = pd.read_csv(OUTPUT)
    else:
        results = df.copy()
        results["probability"] = None
        results["prediction"] = None
        results["correct"] = None
        results["error"] = ""
        results["input_tokens"] = None
        results["output_tokens"] = None
        results["model"] = None

    for i, row in df.iterrows():

        case_id = row["case_id"]

        # Resume safely if already completed.
        existing = results.index[
            results["case_id"] == case_id
        ].tolist()

        if existing:
            idx = existing[0]

            if pd.notna(results.at[idx, "probability"]):
                continue
        else:
            idx = len(results)
            results.loc[idx, :] = None
            results.at[idx, "case_id"] = row["case_id"]
            results.at[idx, "core_id"] = row["core_id"]
            results.at[idx, "domain"] = row["domain"]
            results.at[idx, "language"] = row["language"]
            results.at[idx, "label"] = row["label"]
            results.at[idx, "question"] = row["question"]

        try:
            probability, raw = call_jev(row["question"])

            prediction = int(probability >= 0.5)
            correct = int(prediction == int(row["label"]))

            usage = raw.get("usage", {})

            results.at[idx, "probability"] = probability
            results.at[idx, "prediction"] = prediction
            results.at[idx, "correct"] = correct
            results.at[idx, "error"] = ""
            results.at[idx, "input_tokens"] = usage.get("input_tokens")
            results.at[idx, "output_tokens"] = usage.get("output_tokens")
            results.at[idx, "model"] = raw.get("model")

            print(
                f"[{i+1:03d}/180] "
                f"{case_id} | "
                f"{row['language']} | "
                f"p={probability:.2f} | "
                f"label={int(row['label'])} | "
                f"correct={correct}"
            )

        except Exception as e:
            results.at[idx, "error"] = str(e)

            print(
                f"[{i+1:03d}/180] "
                f"{case_id} | ERROR: {e}"
            )

        results.to_csv(OUTPUT, index=False)

        time.sleep(0.15)

    print("\nRun complete.")

    completed = results["probability"].notna().sum()
    errors = results["error"].fillna("").ne("").sum()

    print(f"Completed: {completed}/180")
    print(f"Errors: {errors}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()