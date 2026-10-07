import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv("/Users/maanikchandela/JEV-Research/jev-decision-control/.env")

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not found in .env")

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

INPUT_PRICE_PER_TOKEN = 42 / 1_000_000_000

# Software-side experiment guard.
# Keep $1 reserve from the $4 research cap.
MAX_SPEND = 3.00

DATASET = Path("experiments/data/e13_decision_type_dataset.csv")

INDEPENDENT_RESULTS = Path(
    "experiments/data/e13_independent_results.csv"
)

BATCH_RESULTS = Path(
    "experiments/data/e13_batch_results.csv"
)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# ============================================================
# API
# ============================================================

def call_jev(state, questions):
    payload = {
        "model": MODEL,
        "state": state,
        "questions": questions,
    }

    response = requests.post(
        ENDPOINT,
        headers=HEADERS,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()
def estimate_cost(response):
    usage = response.get("usage", {})

    input_tokens = usage.get("input_tokens", 0)

    return input_tokens * INPUT_PRICE_PER_TOKEN


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_noul(answer):
    """
    Expected structure:

    answers:
      decision:
        noul: 0.94
    """

    value = answer.get("noul")

    if isinstance(value, (int, float)):
        return float(value)

    raise ValueError(f"Unexpected Noul answer: {answer}")


def parse_choice(answer):
    """
    Expected structure:

    answers:
      decision:
        choice: "true"
        confidence: ...
        probabilities:
          true: ...
          false: ...
          uncertain: ...
    """

    probabilities = answer.get("probabilities")

    if not isinstance(probabilities, dict):
        raise ValueError(
            f"Choice probabilities missing: {answer}"
        )

    if "true" not in probabilities:
        raise ValueError(
            f"'true' missing from Choice probabilities: {answer}"
        )

    return float(probabilities["true"])


def parse_score(answer):
    """
    Score criteria are:

    0 = false
    1 = uncertain
    2 = true

    Therefore P(True) = probabilities["2"].
    """

    probabilities = answer.get("probabilities")

    if not isinstance(probabilities, dict):
        raise ValueError(
            f"Score probabilities missing: {answer}"
        )

    # API may return numeric keys as strings.
    if "2" in probabilities:
        return float(probabilities["2"])

    # Defensive fallback.
    if 2 in probabilities:
        return float(probabilities[2])

    raise ValueError(
        f"Score true-level probability missing: {answer}"
    )


def parse_answer(response, question_name, representation):
    answers = response.get("answers", {})

    if question_name not in answers:
        raise ValueError(
            f"Question '{question_name}' missing. "
            f"Available answers: {list(answers.keys())}"
        )

    answer = answers[question_name]

    if representation == "noul":
        p_true = parse_noul(answer)

    elif representation == "choice":
        p_true = parse_choice(answer)

    elif representation == "score":
        p_true = parse_score(answer)

    else:
        raise ValueError(
            f"Unknown representation: {representation}"
        )

    prediction = int(p_true >= 0.5)

    return p_true, prediction


# ============================================================
# QUESTION BUILDERS
# ============================================================

def build_noul_question(state):
    return {
        "type": "noul",
        "instructions": (
            "Determine whether the statement in the supplied state is true."
        ),
    }

def build_choice_question(state):
    return {
        "type": "choice",
        "instructions": (
            "Classify the truth status of the statement in the supplied state."
        ),
        "criteria": {
            "true": "The statement is true.",
            "false": "The statement is false.",
            "uncertain": (
                "The truth value cannot be determined "
                "from the supplied state."
            ),
        },
    }


def build_score_question(state):
    return {
        "type": "score",
        "instructions": (
            "Rate the truth status of the statement in the supplied state."
        ),
        "criteria": [
            "The statement is false.",
            (
                "The truth value cannot be determined "
                "from the supplied state."
            ),
            "The statement is true.",
        ],
    }

# ============================================================
# COST / SAFETY
# ============================================================

def current_spend():
    total = 0.0

    for path in [INDEPENDENT_RESULTS, BATCH_RESULTS]:
        if path.exists():
            df = pd.read_csv(path)

            if "estimated_cost" in df.columns:
                total += pd.to_numeric(
                    df["estimated_cost"],
                    errors="coerce"
                ).fillna(0).sum()

    return float(total)


def check_budget():
    spend = current_spend()

    if spend >= MAX_SPEND:
        raise RuntimeError(
            f"Budget guard reached: ${spend:.6f}"
        )

    return spend


# ============================================================
# E13-A: INDEPENDENT
# ============================================================

def run_independent(df):
    print("\n" + "=" * 70)
    print("E13-A: INDEPENDENT DECISION TYPES")
    print("=" * 70)

    if INDEPENDENT_RESULTS.exists():
        results = pd.read_csv(INDEPENDENT_RESULTS)
    else:
        results = pd.DataFrame()

    completed = set()

    if not results.empty:
        completed = set(
            zip(
                results["case_id"],
                results["representation"],
            )
        )

    representations = [
        ("noul", build_noul_question),
        ("choice", build_choice_question),
        ("score", build_score_question),
    ]

    total_jobs = len(df) * len(representations)
    job_number = 0

    for _, row in df.iterrows():

        for representation, builder in representations:

            job_number += 1

            key = (
                row["case_id"],
                representation,
            )

            if key in completed:
                print(
                    f"[{job_number}/{total_jobs}] "
                    f"SKIP {row['case_id']} {representation}"
                )
                continue

            spend = check_budget()

            print(
                f"[{job_number}/{total_jobs}] "
                f"{row['case_id']} | {representation} | "
                f"current spend=${spend:.6f}"
            )

            question_name = "decision"

            questions = {
                question_name: builder(row["state"])
            }

            start = time.time()

            try:
                response = call_jev(row["state"], questions)
                latency = time.time() - start

                p_true, prediction = parse_answer(
                    response,
                    question_name,
                    representation,
                )

                usage = response.get("usage", {})

                input_tokens = usage.get(
                    "input_tokens",
                    0
                )

                output_tokens = usage.get(
                    "output_tokens",
                    0
                )

                cost = estimate_cost(response)

                model_returned = response.get(
                    "model",
                    MODEL
                )

                result = {
                    "case_id": row["case_id"],
                    "domain": row["domain"],
                    "difficulty": row["difficulty"],
                    "label": int(row["label"]),
                    "state": row["state"],
                    "representation": representation,
                    "condition": "independent",
                    "probability_true": p_true,
                    "prediction": prediction,
                    "correct": int(
                        prediction == int(row["label"])
                    ),
                    "model": model_returned,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost": cost,
                    "latency_seconds": latency,
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                    "status": "success",
                }

                results = pd.concat(
                    [results, pd.DataFrame([result])],
                    ignore_index=True,
                )

                results.to_csv(
                    INDEPENDENT_RESULTS,
                    index=False,
                )

                print(
                    f"    p_true={p_true:.4f} "
                    f"pred={prediction} "
                    f"label={int(row['label'])} "
                    f"cost=${cost:.8f}"
                )

            except Exception as exc:

                print(
                    f"    ERROR: {type(exc).__name__}: {exc}"
                )

                result = {
                    "case_id": row["case_id"],
                    "domain": row["domain"],
                    "difficulty": row["difficulty"],
                    "label": int(row["label"]),
                    "state": row["state"],
                    "representation": representation,
                    "condition": "independent",
                    "probability_true": None,
                    "prediction": None,
                    "correct": None,
                    "model": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "estimated_cost": 0.0,
                    "latency_seconds": time.time() - start,
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                    "status": f"error: {exc}",
                }

                results = pd.concat(
                    [results, pd.DataFrame([result])],
                    ignore_index=True,
                )

                results.to_csv(
                    INDEPENDENT_RESULTS,
                    index=False,
                )

            time.sleep(0.2)

    print("\nE13-A complete.")


# ============================================================
# E13-B: MIXED-TYPE BATCH
# ============================================================

def run_batch(df):
    print("\n" + "=" * 70)
    print("E13-B: MIXED-TYPE BATCH")
    print("=" * 70)

    if BATCH_RESULTS.exists():
        results = pd.read_csv(BATCH_RESULTS)
    else:
        results = pd.DataFrame()

    completed = set()

    if not results.empty:
        completed = set(
            results["case_id"]
        )

    total_jobs = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        case_id = row["case_id"]

        if case_id in completed:
            print(
                f"[{i}/{total_jobs}] SKIP {case_id}"
            )
            continue

        spend = check_budget()

        print(
            f"[{i}/{total_jobs}] "
            f"{case_id} | mixed batch | "
            f"current spend=${spend:.6f}"
        )

        questions = {
            "noul_decision": build_noul_question(
                row["state"]
            ),
            "choice_decision": build_choice_question(
                row["state"]
            ),
            "score_decision": build_score_question(
                row["state"]
            ),
        }

        start = time.time()

        try:
            response = call_jev(row["state"], questions)
            latency = time.time() - start

            noul_p, noul_pred = parse_answer(
                response,
                "noul_decision",
                "noul",
            )

            choice_p, choice_pred = parse_answer(
                response,
                "choice_decision",
                "choice",
            )

            score_p, score_pred = parse_answer(
                response,
                "score_decision",
                "score",
            )

            usage = response.get("usage", {})

            input_tokens = usage.get(
                "input_tokens",
                0
            )

            output_tokens = usage.get(
                "output_tokens",
                0
            )

            cost = estimate_cost(response)

            model_returned = response.get(
                "model",
                MODEL
            )

            label = int(row["label"])

            result = {
                "case_id": case_id,
                "domain": row["domain"],
                "difficulty": row["difficulty"],
                "label": label,
                "state": row["state"],

                "noul_probability_true": noul_p,
                "choice_probability_true": choice_p,
                "score_probability_true": score_p,

                "noul_prediction": noul_pred,
                "choice_prediction": choice_pred,
                "score_prediction": score_pred,

                "noul_correct": int(
                    noul_pred == label
                ),
                "choice_correct": int(
                    choice_pred == label
                ),
                "score_correct": int(
                    score_pred == label
                ),

                "model": model_returned,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": cost,
                "latency_seconds": latency,

                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),

                "status": "success",
            }

            results = pd.concat(
                [results, pd.DataFrame([result])],
                ignore_index=True,
            )

            results.to_csv(
                BATCH_RESULTS,
                index=False,
            )

            print(
                f"    Noul={noul_p:.4f} "
                f"Choice={choice_p:.4f} "
                f"Score={score_p:.4f} "
                f"cost=${cost:.8f}"
            )

        except Exception as exc:

            print(
                f"    ERROR: {type(exc).__name__}: {exc}"
            )

            result = {
                "case_id": case_id,
                "domain": row["domain"],
                "difficulty": row["difficulty"],
                "label": int(row["label"]),
                "state": row["state"],

                "noul_probability_true": None,
                "choice_probability_true": None,
                "score_probability_true": None,

                "noul_prediction": None,
                "choice_prediction": None,
                "score_prediction": None,

                "noul_correct": None,
                "choice_correct": None,
                "score_correct": None,

                "model": None,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost": 0.0,
                "latency_seconds": time.time() - start,

                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),

                "status": f"error: {exc}",
            }

            results = pd.concat(
                [results, pd.DataFrame([result])],
                ignore_index=True,
            )

            results.to_csv(
                BATCH_RESULTS,
                index=False,
            )

        time.sleep(0.2)

    print("\nE13-B complete.")


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary():
    print("\n" + "=" * 70)
    print("E13 RUN SUMMARY")
    print("=" * 70)

    if INDEPENDENT_RESULTS.exists():
        df = pd.read_csv(INDEPENDENT_RESULTS)

        success = df[
            df["status"] == "success"
        ]

        print("\nE13-A Independent:")
        print(f"Rows: {len(df)}")
        print(f"Successful: {len(success)}")
        print(f"Expected: 180")
        print(
            f"Spend: "
            f"${success['estimated_cost'].sum():.8f}"
        )

    if BATCH_RESULTS.exists():
        df = pd.read_csv(BATCH_RESULTS)

        success = df[
            df["status"] == "success"
        ]

        print("\nE13-B Mixed batch:")
        print(f"Rows: {len(df)}")
        print(f"Successful: {len(success)}")
        print(f"Expected: 60")
        print(
            f"Spend: "
            f"${success['estimated_cost'].sum():.8f}"
        )

    print(
        f"\nCombined recorded spend: "
        f"${current_spend():.8f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("E13 — DECISION-TYPE SENSITIVITY")
    print("=" * 70)

    df = pd.read_csv(DATASET)

    print(f"Dataset cases: {len(df)}")

    assert len(df) == 60

    # Ensure the expected balance is still intact.
    assert df["label"].sum() == 30
    assert (df["label"] == 0).sum() == 30

    print(
        f"Initial spend recorded: "
        f"${current_spend():.8f}"
    )

    # First run independent condition.
    run_independent(df)

    # Then mixed-type batch condition.
    run_batch(df)

    print_summary()


if __name__ == "__main__":
    main()
