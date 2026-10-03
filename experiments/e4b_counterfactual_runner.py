import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "experiments" / "data"

INPUT = DATA / "e4b_counterfactuals.csv"
OUT_JSONL = DATA / "e4b_counterfactual_results.jsonl"
OUT_CSV = DATA / "e4b_counterfactual_results.csv"

load_dotenv(BASE / ".env")

API_KEY = os.getenv("TYPESAFE_API_KEY")

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

PRICE_PER_BILLION_INPUT = 42.0
INTERNAL_LIMIT_USD = 4.0
EST_TOKENS_PER_REQUEST = 500


if not API_KEY:
    raise RuntimeError(
        "TYPESAFE_API_KEY not found in .env"
    )


def estimate_cost(tokens):
    return tokens / 1_000_000_000 * PRICE_PER_BILLION_INPUT


def existing_ids():
    ids = set()

    if OUT_JSONL.exists():
        with OUT_JSONL.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        ids.add(
                            json.loads(line)["variant_id"]
                        )
                    except Exception:
                        pass

    return ids


def call_jev(state, question):
    payload = {
        "model": MODEL,
        "state": state,
        "questions": {
            "decision": {
                "type": "noul",
                "instructions": question,
                "criteria": {
                    "true": "The statement in the question is true.",
                    "false": "The statement in the question is false.",
                },
            }
        },
    }

    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT}"
        )

    with INPUT.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    completed = existing_ids()

    remaining = [
        row
        for row in rows
        if row["variant_id"] not in completed
    ]

    current_spend = 0.0

    if OUT_JSONL.exists():
        with OUT_JSONL.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        current_spend += float(
                            json.loads(line).get(
                                "estimated_cost_usd",
                                0
                            )
                        )
                    except Exception:
                        pass

    print("=" * 60)
    print("E4b COUNTERFACTUAL STABILITY RUN")
    print("=" * 60)

    print(f"Total variants: {len(rows)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print(
        f"Current estimated spend: "
        f"${current_spend:.8f}"
    )

    max_additional = estimate_cost(
        len(remaining) * EST_TOKENS_PER_REQUEST
    )

    print(
        f"Estimated maximum additional spend: "
        f"${max_additional:.8f}"
    )

    print()

    fields = [
        "variant_id",
        "candidate_id",
        "baseline_probability",
        "domain",
        "perturbation_type",
        "state",
        "question",
        "true_label",
        "jev_probability",
        "predicted_label",
        "correct",
        "absolute_delta_p",
        "decision_flip",
        "model",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "http_status",
        "timestamp_utc",
    ]

    for i, row in enumerate(remaining, 1):

        projected = (
            current_spend
            + estimate_cost(EST_TOKENS_PER_REQUEST)
        )

        if projected > INTERNAL_LIMIT_USD:
            raise RuntimeError(
                "Internal research budget would be exceeded."
            )

        print(
            f"[{i}/{len(remaining)}] "
            f"{row['variant_id']} "
            f"({row['perturbation_type']})..."
        )

        try:

            result = call_jev(
                row["state"],
                row["question"]
            )

            answer = result["answers"]["decision"]

            probability = float(
                answer["noul"]
            )

            baseline = float(
                row["baseline_probability"]
            )

            predicted = int(
                probability >= 0.5
            )

            baseline_prediction = int(
                baseline >= 0.5
            )

            true_label = int(
                row["true_label"]
            )

            correct = int(
                predicted == true_label
            )

            delta = abs(
                probability - baseline
            )

            flip = int(
                predicted != baseline_prediction
            )

            usage = result.get(
                "usage",
                {}
            )

            input_tokens = int(
                usage.get(
                    "input_tokens",
                    0
                )
            )

            output_tokens = int(
                usage.get(
                    "output_tokens",
                    0
                )
            )

            cost = estimate_cost(
                input_tokens
            )

            record = {
                "variant_id": row["variant_id"],
                "candidate_id": row["candidate_id"],
                "baseline_probability": baseline,
                "domain": row["domain"],
                "perturbation_type": row[
                    "perturbation_type"
                ],
                "state": row["state"],
                "question": row["question"],
                "true_label": true_label,
                "jev_probability": probability,
                "predicted_label": predicted,
                "correct": correct,
                "absolute_delta_p": delta,
                "decision_flip": flip,
                "model": result.get(
                    "model",
                    MODEL
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": cost,
                "http_status": 200,
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            with OUT_JSONL.open(
                "a",
                encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

            current_spend += cost

            print(
                f"    baseline = {baseline:.2f}"
            )

            print(
                f"    variant  = {probability:.2f}"
            )

            print(
                f"    |delta|  = {delta:.2f}"
            )

            print(
                f"    flip     = {bool(flip)}"
            )

            print(
                f"    correct  = {bool(correct)}"
            )

            print(
                f"    cost     = ${cost:.8f}"
            )

        except Exception as e:

            print(
                f"    ERROR: {e}"
            )

        time.sleep(0.15)

    records = []

    if OUT_JSONL.exists():
        with OUT_JSONL.open(
            encoding="utf-8"
        ) as f:
            records = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(records)

    print()
    print("=" * 60)
    print("E4b COUNTERFACTUAL RUN COMPLETE")
    print("=" * 60)

    print(
        f"Results: {OUT_CSV}"
    )

    print(
        f"Estimated total spend: "
        f"${current_spend:.8f}"
    )


if __name__ == "__main__":
    main()
