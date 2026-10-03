import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "experiments" / "data"

INPUT_FILE = DATA_DIR / "e4_counterfactuals_final.csv"
OUTPUT_JSONL = DATA_DIR / "e4_results.jsonl"
OUTPUT_CSV = DATA_DIR / "e4_results.csv"


# ============================================================
# JEV API
# ============================================================

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


# ============================================================
# BUDGET SAFETY
# ============================================================

INTERNAL_BUDGET_USD = 4.00
PRICE_PER_BILLION_INPUT_TOKENS = 42.0
ESTIMATED_INPUT_TOKENS = 500


def token_cost(input_tokens):
    return (
        input_tokens / 1_000_000_000
    ) * PRICE_PER_BILLION_INPUT_TOKENS


def get_existing_spend():
    total = 0.0

    if not OUTPUT_JSONL.exists():
        return total

    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                total += float(
                    record.get("estimated_cost_usd", 0.0)
                )
            except (json.JSONDecodeError, ValueError):
                continue

    return total


# ============================================================
# API KEY
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        f"TYPESAFE_API_KEY is missing.\n"
        f"Expected .env file at:\n{ENV_FILE}"
    )


# ============================================================
# DATA
# ============================================================

def load_counterfactuals():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"E4 dataset not found:\n{INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_completed_ids():
    completed = set()

    if not OUTPUT_JSONL.exists():
        return completed

    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                cid = record.get("counterfactual_id")

                if cid:
                    completed.add(cid)

            except json.JSONDecodeError:
                continue

    return completed


# ============================================================
# JEV REQUEST
# ============================================================

def query_jev(row):

    payload = {
        "model": MODEL,
        "state": row["state"],
        "questions": {
            row["counterfactual_id"]: {
                "type": "noul",
                "instructions": row["question"],
                "criteria": {
                    "true": "The statement in the question is true.",
                    "false": "The statement in the question is false.",
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
        timeout=30,
    )

    return response


# ============================================================
# SAVE RESULT
# ============================================================

def save_record(record):

    # JSONL
    with open(
        OUTPUT_JSONL,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    # CSV
    file_exists = OUTPUT_CSV.exists()

    fieldnames = [
        "experiment_id",
        "counterfactual_id",
        "original_id",
        "perturbation_type",
        "state",
        "question",
        "true_label",
        "domain",
        "difficulty",
        "jev_probability",
        "predicted_label",
        "model",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "http_status",
        "timestamp_utc",
    ]

    with open(
        OUTPUT_CSV,
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)


# ============================================================
# RUN
# ============================================================

def run(start, end):

    rows = load_counterfactuals()
    completed = load_completed_ids()

    selected_rows = rows[start - 1:end]

    print()
    print("E4 COUNTERFACTUAL STABILITY")
    print("=" * 50)
    print(f"Dataset rows: {len(rows)}")
    print(f"Selected rows: {start}-{end}")
    print(f"Already completed: {len(completed)}")
    print()

    current_spend = get_existing_spend()

    print(
        f"Current E4 estimated spend: "
        f"${current_spend:.8f}"
    )

    print(
        f"Internal research limit: "
        f"${INTERNAL_BUDGET_USD:.2f}"
    )

    estimated_total = (
        current_spend
        + len(selected_rows)
        * token_cost(ESTIMATED_INPUT_TOKENS)
    )

    print(
        f"Estimated maximum after this run: "
        f"${estimated_total:.8f}"
    )

    if estimated_total > INTERNAL_BUDGET_USD:
        raise RuntimeError(
            "Budget guard stopped the experiment. "
            "The internal $4 research limit would be exceeded."
        )

    print()

    for index, row in enumerate(selected_rows, start=start):

        counterfactual_id = row["counterfactual_id"]

        if counterfactual_id in completed:
            print(
                f"[{index}] SKIP "
                f"{counterfactual_id} "
                f"(already completed)"
            )
            continue

        print(
            f"[{index}] Running "
            f"{counterfactual_id} "
            f"({row['perturbation_type']})..."
        )

        response = query_jev(row)

        try:
            result = response.json()
        except ValueError:
            result = {
                "raw_response": response.text
            }

        if response.status_code != 200:
            print(
                f"    ERROR: HTTP {response.status_code}"
            )
            print(
                f"    Response: {response.text[:500]}"
            )
            continue

        answer = (
            result
            .get("answers", {})
            .get(counterfactual_id, {})
        )

        probability = answer.get("noul")

        if probability is None:
            print("    ERROR: No JEV probability returned.")
            continue

        probability = float(probability)

        predicted_label = (
            1 if probability >= 0.5 else 0
        )

        usage = result.get("usage", {})

        input_tokens = usage.get(
            "input_tokens",
            0,
        )

        output_tokens = usage.get(
            "output_tokens",
            0,
        )

        actual_cost = token_cost(
            input_tokens
        )

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        record = {
            "experiment_id": "E4",
            "counterfactual_id": counterfactual_id,
            "original_id": row["original_id"],
            "perturbation_type": row["perturbation_type"],
            "state": row["state"],
            "question": row["question"],
            "true_label": int(row["true_label"]),
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "jev_probability": probability,
            "predicted_label": predicted_label,
            "model": result.get("model"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": actual_cost,
            "http_status": response.status_code,
            "timestamp_utc": timestamp,
        }

        save_record(record)

        print(
            f"    probability = {probability:.4f}"
        )

        print(
            f"    prediction  = {predicted_label}"
        )

        print(
            f"    cost        = ${actual_cost:.8f}"
        )

    print()
    print("E4 run complete.")
    print(f"Results: {OUTPUT_CSV}")


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Run E4 counterfactual stability experiment."
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Starting dataset row (1-indexed).",
    )

    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Ending dataset row (inclusive).",
    )

    args = parser.parse_args()

    if args.start < 1:
        raise ValueError("--start must be >= 1")

    if args.end < args.start:
        raise ValueError(
            "--end must be >= --start"
        )

    run(
        args.start,
        args.end,
    )


if __name__ == "__main__":
    main()