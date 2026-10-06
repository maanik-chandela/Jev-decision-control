import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

DATASET_PATH = "experiments/data/e11_decision_decomposition_dataset.csv"
RESULTS_PATH = "experiments/data/e11_decision_decomposition_results.csv"

SLEEP_SECONDS = 0.15
MAX_RETRIES = 5


# ============================================================
# LOAD API KEY
# ============================================================

PROJECT_ROOT = Path.cwd()
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "TYPESAFE_API_KEY was not found in the project .env file."
    )


HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# ============================================================
# API FUNCTION
# ============================================================

def call_jev(state, question):
    """
    Ask JEV one binary decision question.

    Returns:
        probability, predicted_label, raw_response
    """

    payload = {
        "model": MODEL,
        "state": state,
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

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()

                probability = float(
                    data["answers"]["q1"]["noul"]
                )

                prediction = int(probability >= 0.5)

                return probability, prediction, data

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        except Exception as exc:
            last_error = str(exc)

        if attempt < MAX_RETRIES:
            wait_time = 2 ** (attempt - 1)

            print(
                f"    Retry {attempt}/{MAX_RETRIES - 1} "
                f"after error: {last_error}"
            )

            time.sleep(wait_time)

    raise RuntimeError(last_error)


# ============================================================
# HELPER
# ============================================================

def make_row(
    case_id,
    domain,
    difficulty,
    label,
    condition,
    question_id,
    probability,
    prediction,
):
    return {
        "case_id": case_id,
        "domain": domain,
        "difficulty": difficulty,
        "label": label,
        "condition": condition,
        "question_id": question_id,
        "probability": probability,
        "prediction": prediction,
        "correct": int(prediction == label),
    }


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results():

    if not os.path.exists(RESULTS_PATH):
        return pd.DataFrame(
            columns=[
                "case_id",
                "domain",
                "difficulty",
                "label",
                "condition",
                "question_id",
                "probability",
                "prediction",
                "correct",
            ]
        )

    df = pd.read_csv(RESULTS_PATH)

    print(
        f"Existing results found: {len(df)} rows"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("E11 DECISION DECOMPOSITION RUNNER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset cases: {len(df)}")

    assert len(df) == 60
    assert df["case_id"].nunique() == 60

    # --------------------------------------------------------
    # Load previous results
    # --------------------------------------------------------

    results = load_existing_results()

    completed = set(
        zip(
            results["case_id"],
            results["condition"],
        )
    )

    print(
        f"Completed case-condition pairs: "
        f"{len(completed)}"
    )

    # --------------------------------------------------------
    # API call counter
    # --------------------------------------------------------

    new_calls = 0
    skipped = 0
    failures = 0

    # --------------------------------------------------------
    # Process cases
    # --------------------------------------------------------

    for index, row in df.iterrows():

        case_id = row["case_id"]

        print()
        print(
            f"[{index + 1}/60] "
            f"{case_id} | "
            f"{row['domain']} | "
            f"{row['difficulty']} | "
            f"label={row['label']}"
        )

        # ====================================================
        # CONDITION 1: DIRECT
        # ====================================================

        condition = "direct"

        if (case_id, condition) in completed:

            print("  DIRECT: already completed")

        else:

            print("  DIRECT: calling JEV...")

            try:

                probability, prediction, _ = call_jev(
                    state=row["compound_state"],
                    question=row["compound_question"],
                )

                new_row = make_row(
                    case_id=case_id,
                    domain=row["domain"],
                    difficulty=row["difficulty"],
                    label=int(row["label"]),
                    condition=condition,
                    question_id="compound",
                    probability=probability,
                    prediction=prediction,
                )

                results = pd.concat(
                    [results, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

                results.to_csv(
                    RESULTS_PATH,
                    index=False,
                )

                completed.add((case_id, condition))
                new_calls += 1

                print(
                    f"    p={probability:.4f} "
                    f"pred={prediction} "
                    f"correct={new_row['correct']}"
                )

            except Exception as exc:

                failures += 1

                print(
                    f"    FAILED: {exc}"
                )

        time.sleep(SLEEP_SECONDS)

        # ====================================================
        # CONDITION 2: DECOMPOSED
        #
        # Two separate JEV calls:
        # A and B
        # ====================================================

        condition = "decomposed"

        if (case_id, condition) in completed:

            print("  DECOMPOSED: already completed")

        else:

            print("  DECOMPOSED: evaluating A and B...")

            try:

                # ----------------------------
                # A
                # ----------------------------

                p_a, pred_a, _ = call_jev(
                    state=row["A_state"],
                    question=row["A_question"],
                )

                time.sleep(SLEEP_SECONDS)

                # ----------------------------
                # B
                # ----------------------------

                p_b, pred_b, _ = call_jev(
                    state=row["B_state"],
                    question=row["B_question"],
                )

                # ------------------------------------------------
                # Binary decomposition decision:
                #
                # Compound = A AND B
                # ------------------------------------------------

                prediction = int(
                    pred_a == 1 and pred_b == 1
                )

                # We store the two probabilities separately.
                # We DO NOT multiply p_a * p_b.
                #
                # For the main decomposition probability proxy,
                # use the conservative minimum.
                probability = min(p_a, p_b)

                new_row = make_row(
                    case_id=case_id,
                    domain=row["domain"],
                    difficulty=row["difficulty"],
                    label=int(row["label"]),
                    condition=condition,
                    question_id="A_AND_B",
                    probability=probability,
                    prediction=prediction,
                )

                # Add component probabilities as extra columns
                new_row["p_A"] = p_a
                new_row["p_B"] = p_b
                new_row["prediction_A"] = pred_a
                new_row["prediction_B"] = pred_b
                new_row["correct_A"] = int(
                    pred_a == int(row["A_label"])
                )
                new_row["correct_B"] = int(
                    pred_b == int(row["B_label"])
                )

                results = pd.concat(
                    [results, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

                results.to_csv(
                    RESULTS_PATH,
                    index=False,
                )

                completed.add((case_id, condition))
                new_calls += 2

                print(
                    f"    A: p={p_a:.4f} pred={pred_a}"
                )

                print(
                    f"    B: p={p_b:.4f} pred={pred_b}"
                )

                print(
                    f"    A AND B: "
                    f"pred={prediction} "
                    f"correct={new_row['correct']}"
                )

            except Exception as exc:

                failures += 1

                print(
                    f"    FAILED: {exc}"
                )

        time.sleep(SLEEP_SECONDS)

        # ====================================================
        # CONDITION 3: BATCHED
        #
        # Target + two self-contained distractors
        # ====================================================

        condition = "batched"

        if (case_id, condition) in completed:

            print("  BATCHED: already completed")

        else:

            print("  BATCHED: calling JEV...")

            try:

                payload = {
                    "model": MODEL,
                    "state": row["compound_state"],
                    "questions": {
                        "target": {
                            "type": "noul",
                            "instructions": row[
                                "compound_question"
                            ],
                            "criteria": {
                                "true": "The statement is true.",
                                "false": "The statement is false.",
                            },
                        },
                        "distractor_1": {
                            "type": "noul",
                            "instructions": (
                                "A triangle has three interior "
                                "angles whose measures sum to "
                                "180 degrees. Is this statement true?"
                            ),
                            "criteria": {
                                "true": "The statement is true.",
                                "false": "The statement is false.",
                            },
                        },
                        "distractor_2": {
                            "type": "noul",
                            "instructions": (
                                "The number 2 is added to itself. "
                                "Is the resulting value 4?"
                            ),
                            "criteria": {
                                "true": "The statement is true.",
                                "false": "The statement is false.",
                            },
                        },
                    },
                }

                last_error = None

                for attempt in range(1, MAX_RETRIES + 1):

                    try:

                        response = requests.post(
                            API_URL,
                            headers=HEADERS,
                            json=payload,
                            timeout=60,
                        )

                        if response.status_code == 200:

                            data = response.json()

                            target_probability = float(
                                data["answers"]["target"]["noul"]
                            )

                            distractor_1_probability = float(
                                data["answers"]["distractor_1"]["noul"]
                            )

                            distractor_2_probability = float(
                                data["answers"]["distractor_2"]["noul"]
                            )

                            break

                        last_error = (
                            f"HTTP {response.status_code}: "
                            f"{response.text[:500]}"
                        )

                    except Exception as exc:

                        last_error = str(exc)

                    if attempt < MAX_RETRIES:

                        wait_time = 2 ** (attempt - 1)

                        print(
                            f"    Batch retry "
                            f"{attempt}/{MAX_RETRIES - 1}"
                        )

                        time.sleep(wait_time)

                else:

                    raise RuntimeError(last_error)

                target_prediction = int(
                    target_probability >= 0.5
                )

                new_row = make_row(
                    case_id=case_id,
                    domain=row["domain"],
                    difficulty=row["difficulty"],
                    label=int(row["label"]),
                    condition=condition,
                    question_id="target",
                    probability=target_probability,
                    prediction=target_prediction,
                )

                new_row["distractor_1_probability"] = (
                    distractor_1_probability
                )

                new_row["distractor_2_probability"] = (
                    distractor_2_probability
                )

                results = pd.concat(
                    [results, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

                results.to_csv(
                    RESULTS_PATH,
                    index=False,
                )

                completed.add((case_id, condition))
                new_calls += 1

                print(
                    f"    target p={target_probability:.4f} "
                    f"pred={target_prediction} "
                    f"correct={new_row['correct']}"
                )

                print(
                    f"    distractor1 p="
                    f"{distractor_1_probability:.4f}"
                )

                print(
                    f"    distractor2 p="
                    f"{distractor_2_probability:.4f}"
                )

            except Exception as exc:

                failures += 1

                print(
                    f"    FAILED: {exc}"
                )

        time.sleep(SLEEP_SECONDS)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("E11 RUN SUMMARY")
    print("=" * 70)

    print(
        f"New API calls made: {new_calls}"
    )

    print(
        f"Skipped completed conditions: {skipped}"
    )

    print(
        f"Failures: {failures}"
    )

    print(
        f"Result rows saved: {len(results)}"
    )

    print(
        f"Results file: {RESULTS_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

