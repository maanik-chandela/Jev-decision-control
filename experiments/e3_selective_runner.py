import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")
API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

INPUT_FILE = "experiments/data/e3_selective_prediction_dataset.csv"
OUTPUT_FILE = "experiments/data/e3_selective_prediction_results.csv"

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")


def call_jev(state, question):
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

    probability = float(answer["noul"])

    usage = data.get("usage", {})

    return (
        probability,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} cases.")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    results = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Running E3"
    ):
        original_id = row["original_id"]
        domain = row["domain"]
        difficulty = row["difficulty"]
        state = row["state"]
        question = row["question"]
        label = int(row["label"])

        try:
            probability, input_tokens, output_tokens = call_jev(
                state,
                question,
            )

            prediction = int(probability >= 0.5)
            correct = int(prediction == label)

            error = ""

        except Exception as e:
            probability = None
            input_tokens = 0
            output_tokens = 0
            prediction = None
            correct = None
            error = str(e)

        results.append({
            "original_id": original_id,
            "domain": domain,
            "difficulty": difficulty,
            "label": label,
            "probability": probability,
            "prediction": prediction,
            "correct": correct,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": error,
        })

        # Small delay to avoid unnecessarily aggressive requests.
        time.sleep(0.1)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    successful = results_df["probability"].notna().sum()
    failed = len(results_df) - successful

    print()
    print("=" * 60)
    print("E3 RUN COMPLETE")
    print("=" * 60)
    print(f"Total cases: {len(results_df)}")
    print(f"Successful:  {successful}")
    print(f"Failed:      {failed}")

    if successful > 0:
        valid = results_df[results_df["probability"].notna()]

        accuracy = valid["correct"].mean()

        total_input = valid["input_tokens"].sum()
        total_output = valid["output_tokens"].sum()

        estimated_cost = total_input * 42 / 1_000_000_000

        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Input tokens:  {total_input:,}")
        print(f"Output tokens: {total_output:,}")
        print(f"Estimated cost: ${estimated_cost:.8f}")

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
