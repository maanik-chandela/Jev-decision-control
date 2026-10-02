import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


# ========================================================
# PROJECT PATHS
# ========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "experiments" / "data"

INPUT_FILE = DATA_DIR / "pilot_questions.csv"
OUTPUT_JSONL = DATA_DIR / "pilot_results.jsonl"
OUTPUT_CSV = DATA_DIR / "pilot_results.csv"


# ========================================================
# JEV API CONFIGURATION
# ========================================================

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


# ========================================================
# RESEARCH BUDGET SAFETY
# ========================================================

# User's actual maximum budget is $5.
# We intentionally keep a $1 safety reserve.
INTERNAL_BUDGET_USD = 4.00

# Current TypeSafe pricing:
# $42 per 1 billion input tokens
PRICE_PER_BILLION_INPUT_TOKENS = 42.0

# Conservative estimate used BEFORE making an API request.
ESTIMATED_INPUT_TOKENS = 500


# ========================================================
# LOAD API KEY FROM PROJECT ROOT .env
# ========================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        f"TYPESAFE_API_KEY is missing.\n"
        f"Expected .env file at:\n{ENV_FILE}"
    )


# ========================================================
# HELPER FUNCTIONS
# ========================================================

def token_cost(input_tokens):
    """
    Calculate estimated JEV input cost.

    Price = $42 per 1 billion input tokens.
    """

    return (input_tokens / 1_000_000_000) * PRICE_PER_BILLION_INPUT_TOKENS


def get_existing_spend():
    """
    Read previous pilot results and calculate cumulative spend.
    """

    if not OUTPUT_JSONL.exists():
        return 0.0

    total = 0.0

    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                total += float(
                    record.get("estimated_cost_usd", 0.0)
                )
            except json.JSONDecodeError:
                continue

    return total


def load_questions():
    """
    Load pilot questions from CSV.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Pilot dataset not found:\n{INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    if not questions:
        raise ValueError("pilot_questions.csv is empty.")

    return questions


def load_completed_question_ids():
    """
    Find questions that have already been run.

    This prevents accidental duplicate API calls.
    """

    completed = set()

    if not OUTPUT_JSONL.exists():
        return completed

    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                question_id = record.get("question_id")

                if question_id:
                    completed.add(question_id)

            except json.JSONDecodeError:
                continue

    return completed


def query_jev(row):
    """
    Send one research question to JEV.
    """

    question_id = row["id"]
    state = row["state"]
    question = row["question"]

    payload = {
        "model": MODEL,
        "state": state,
        "questions": {
            question_id: {
                "type": "noul",
                "instructions": question,
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


def save_record(record):
    """
    Save one research result to JSONL and CSV.
    """

    # ----------------------------------------------------
    # JSONL
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # CSV
    # ----------------------------------------------------

    file_exists = OUTPUT_CSV.exists()

    fieldnames = [
        "experiment_id",
        "question_id",
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


# ========================================================
# MAIN
# ========================================================

def main():

    print("=" * 42)
    print("JEV PILOT RUNNER")
    print("=" * 42)

    # ----------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------

    questions = load_questions()

    print(f"Total questions available: {len(questions)}")

    # ----------------------------------------------------
    # COMMAND-LINE ARGUMENTS
    # ----------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Run a controlled batch of the JEV pilot dataset."
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting question number, inclusive.",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=5,
        help="Ending question number, inclusive.",
    )

    args = parser.parse_args()

    # ----------------------------------------------------
    # VALIDATE RANGE
    # ----------------------------------------------------

    if args.start < 1:
        raise ValueError(
            "--start must be at least 1."
        )

    if args.end < args.start:
        raise ValueError(
            "--end must be greater than or equal to --start."
        )

    if args.end > len(questions):
        raise ValueError(
            f"--end cannot exceed {len(questions)}."
        )

    # Python slicing:
    # --start 6 --end 10
    # becomes questions[5:10]
    questions = questions[
        args.start - 1 : args.end
    ]

    print(
        f"Questions in this run: "
        f"P{args.start:03d} to P{args.end:03d}"
    )

    # ----------------------------------------------------
    # CURRENT SPEND
    # ----------------------------------------------------

    existing_spend = get_existing_spend()

    print(
        f"Internal budget: "
        f"${INTERNAL_BUDGET_USD:.2f}"
    )

    print(
        f"Current estimated spend: "
        f"${existing_spend:.8f}"
    )

    print()

    # ----------------------------------------------------
    # COMPLETED QUESTIONS
    # ----------------------------------------------------

    completed_ids = load_completed_question_ids()

    if completed_ids:
        print(
            f"Previously completed questions: "
            f"{len(completed_ids)}"
        )

    print()

    # ----------------------------------------------------
    # RUN QUESTIONS
    # ----------------------------------------------------

    total_new_spend = 0.0

    for row in questions:

        question_id = row["id"]

        # ------------------------------------------------
        # DUPLICATE SAFETY
        # ------------------------------------------------

        if question_id in completed_ids:

            print(
                f"{question_id} SKIPPED "
                f"(already completed)"
            )

            continue

        # ------------------------------------------------
        # PRE-REQUEST BUDGET CHECK
        # ------------------------------------------------

        estimated_request_cost = token_cost(
            ESTIMATED_INPUT_TOKENS
        )

        current_total = (
            existing_spend
            + total_new_spend
        )

        if (
            current_total
            + estimated_request_cost
            > INTERNAL_BUDGET_USD
        ):

            print()
            print("=" * 42)
            print("BUDGET SAFETY STOP")
            print("=" * 42)

            print(
                f"Current estimated spend: "
                f"${current_total:.8f}"
            )

            print(
                f"Estimated next request: "
                f"${estimated_request_cost:.8f}"
            )

            print(
                f"Internal limit: "
                f"${INTERNAL_BUDGET_USD:.2f}"
            )

            print(
                "No further API calls were made."
            )

            break

        # ------------------------------------------------
        # API REQUEST
        # ------------------------------------------------

        try:

            response = query_jev(row)

        except requests.RequestException as e:

            print(
                f"{question_id} REQUEST ERROR: "
                f"{e}"
            )

            continue

        print(
            f"{question_id} HTTP "
            f"{response.status_code}",
            end="",
        )

        # ------------------------------------------------
        # HANDLE API ERROR
        # ------------------------------------------------

        if response.status_code != 200:

            print()

            print(
                "JEV API error response:"
            )

            print(response.text)

            continue

        # ------------------------------------------------
        # PARSE RESPONSE
        # ------------------------------------------------

        try:

            result = response.json()

        except ValueError:

            print()

            print(
                "ERROR: API response was not valid JSON."
            )

            print(response.text)

            continue

        try:

            answer = result[
                "answers"
            ][
                question_id
            ]

            probability = float(
                answer["noul"]
            )

            usage = result.get(
                "usage",
                {},
            )

            input_tokens = int(
                usage.get(
                    "input_tokens",
                    0,
                )
            )

            output_tokens = int(
                usage.get(
                    "output_tokens",
                    0,
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as e:

            print()

            print(
                "ERROR: Unexpected JEV response format."
            )

            print(
                f"Details: {e}"
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

            continue

        # ------------------------------------------------
        # PREDICTION
        # ------------------------------------------------

        predicted_label = (
            1
            if probability >= 0.5
            else 0
        )

        true_label = int(
            row["true_label"]
        )

        # ------------------------------------------------
        # COST
        # ------------------------------------------------

        actual_cost = token_cost(
            input_tokens
        )

        total_new_spend += actual_cost

        # ------------------------------------------------
        # RESEARCH RECORD
        # ------------------------------------------------

        experiment_id = (
            f"JEV-PILOT-{question_id}"
        )

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        record = {

            "experiment_id":
                experiment_id,

            "question_id":
                question_id,

            "state":
                row["state"],

            "question":
                row["question"],

            "true_label":
                true_label,

            "domain":
                row["domain"],

            "difficulty":
                row["difficulty"],

            "jev_probability":
                probability,

            "predicted_label":
                predicted_label,

            "model":
                result.get(
                    "model",
                    MODEL,
                ),

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,

            "estimated_cost_usd":
                actual_cost,

            "timestamp_utc":
                timestamp,
        }

        # ------------------------------------------------
        # SAVE
        # ------------------------------------------------

        save_record(record)

        # ------------------------------------------------
        # PRINT RESULT
        # ------------------------------------------------

        print(
            f", probability {probability:.4f}, "
            f"predicted {predicted_label}, "
            f"ground truth {true_label}, "
            f"input {input_tokens}, "
            f"output {output_tokens}, "
            f"cost ${actual_cost:.8f}"
        )

    # ----------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------

    final_spend = (
        existing_spend
        + total_new_spend
    )

    print()
    print("=" * 42)

    print(
        f"New estimated spend: "
        f"${total_new_spend:.8f}"
    )

    print(
        f"Total estimated spend: "
        f"${final_spend:.8f}"
    )

    print(
        f"Internal research limit: "
        f"${INTERNAL_BUDGET_USD:.2f}"
    )

    print()
    print("Output files:")

    print(OUTPUT_JSONL)
    print(OUTPUT_CSV)

    print("=" * 42)


# ========================================================
# ENTRY POINT
# ========================================================

if __name__ == "__main__":
    main()