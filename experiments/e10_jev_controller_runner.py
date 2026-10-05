#!/usr/bin/env python3

"""
E10 — JEV Controller Runner

Uses JEV as a controller for the Qwen3:8b Stage-1 answer.

For each case:
    Question
        ↓
    Qwen3:8b answer
        ↓
    JEV evaluates:
        "Is the Stage-1 answer correct?"
        ↓
    JEV probability that Stage-1 is correct

Input:
    experiments/data/e10_computation_allocation_dataset.csv
    experiments/data/e10_base_results.csv

Output:
    experiments/data/e10_jev_controller_results.csv
"""

from pathlib import Path
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "e10_computation_allocation_dataset.csv"
)

BASE_RESULTS_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "e10_base_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "e10_jev_controller_results.csv"
)

ENV_FILE = PROJECT_ROOT / ".env"


# ============================================================
# JEV CONFIGURATION
# ============================================================

API_URL = "https://api.typesafe.ai/v1/systemone"

MODEL = "jev-latest"

REQUEST_TIMEOUT = 120

MAX_RETRIES = 3

RETRY_SLEEP_SECONDS = 2


# ============================================================
# API KEY
# ============================================================

load_dotenv(ENV_FILE)

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "TYPESAFE_API_KEY was not found in .env"
    )


# ============================================================
# BUILD JEV REQUEST
# ============================================================

def build_payload(question, stage1_answer):

    state = (
        "Original question:\n"
        f"{question}\n\n"
        "Stage-1 model answer:\n"
        f"{stage1_answer}"
    )

    return {
        "model": MODEL,
        "state": state,
        "questions": {
            "stage1_answer_correct": {
                "type": "noul",
                "instructions": (
                    "Is the Stage-1 model's answer to "
                    "the original question correct?"
                ),
                "criteria": {
                    "true": (
                        "The Stage-1 answer is correct."
                    ),
                    "false": (
                        "The Stage-1 answer is incorrect."
                    ),
                },
            }
        },
    }


# ============================================================
# CALL JEV
# ============================================================

def call_jev(question, stage1_answer):

    payload = build_payload(
        question,
        stage1_answer
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        start = time.perf_counter()

        try:

            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            latency = (
                time.perf_counter() - start
            )

            response.raise_for_status()

            data = response.json()

            answer = (
                data
                .get("answers", {})
                .get("stage1_answer_correct")
            )

            if answer is None:
                raise ValueError(
                    "Missing "
                    "answers.stage1_answer_correct"
                )

            probability = float(
                answer["noul"]
            )

            prediction = int(
                probability >= 0.5
            )

            usage = data.get(
                "usage",
                {}
            )

            input_tokens = usage.get(
                "input_tokens",
                usage.get("input")
            )

            output_tokens = usage.get(
                "output_tokens",
                usage.get("output")
            )

            returned_model = data.get(
                "model"
            )

            return {
                "jev_probability_correct":
                    probability,

                "jev_prediction_correct":
                    prediction,

                "latency_seconds":
                    latency,

                "input_tokens":
                    input_tokens,

                "output_tokens":
                    output_tokens,

                "jev_model":
                    returned_model,

                "error":
                    None,
            }

        except Exception as exc:

            last_error = repr(exc)

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_SLEEP_SECONDS * attempt
                )

    return {
        "jev_probability_correct":
            None,

        "jev_prediction_correct":
            None,

        "latency_seconds":
            0,

        "input_tokens":
            None,

        "output_tokens":
            None,

        "jev_model":
            None,

        "error":
            last_error,
    }


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results():

    if not OUTPUT_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            OUTPUT_FILE
        )

    except Exception:

        print(
            "Warning: could not read existing "
            "JEV results. Starting fresh."
        )

        return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("E10 JEV CONTROLLER")
    print("=" * 60)


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    if not BASE_RESULTS_FILE.exists():

        raise FileNotFoundError(
            "Qwen baseline results not found.\n\n"
            "Run:\n"
            "python experiments/e10_base_runner.py"
        )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    dataset = pd.read_csv(
        DATASET_FILE
    )

    baseline = pd.read_csv(
        BASE_RESULTS_FILE
    )


    # --------------------------------------------------------
    # DATASET VALIDATION
    # --------------------------------------------------------

    assert len(dataset) == 150

    assert dataset["case_id"].is_unique

    assert set(
        dataset["label"].unique()
    ) == {0, 1}


    # --------------------------------------------------------
    # BASELINE VALIDATION
    # --------------------------------------------------------

    required_columns = {
        "case_id",
        "prediction",
        "correct",
        "raw_response",
    }

    missing = (
        required_columns
        - set(baseline.columns)
    )

    if missing:

        raise ValueError(
            "Baseline is missing columns: "
            f"{sorted(missing)}"
        )


    dataset_ids = set(
        dataset["case_id"]
        .astype(str)
    )

    baseline_ids = set(
        baseline["case_id"]
        .astype(str)
    )

    missing_baseline = (
        dataset_ids - baseline_ids
    )

    if missing_baseline:

        raise ValueError(
            "Missing Qwen results for:\n"
            f"{sorted(missing_baseline)}"
        )


    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    baseline_merge = baseline[
        [
            "case_id",
            "prediction",
            "correct",
            "raw_response",
        ]
    ].copy()

    baseline_merge["case_id"] = (
        baseline_merge["case_id"]
        .astype(str)
    )

    merged = dataset.copy()

    merged["case_id"] = (
        merged["case_id"]
        .astype(str)
    )

    merged = merged.merge(
        baseline_merge,
        on="case_id",
        how="left",
        validate="one_to_one",
    )


    # --------------------------------------------------------
    # CHECK PREDICTIONS
    # --------------------------------------------------------

    if merged["prediction"].isna().any():

        missing_rows = merged[
            merged["prediction"].isna()
        ]

        raise ValueError(
            "Some Qwen predictions are missing:\n"
            + missing_rows[
                ["case_id", "question"]
            ].to_string(index=False)
        )


    # --------------------------------------------------------
    # EXISTING JEV RESULTS
    # --------------------------------------------------------

    existing = load_existing_results()

    completed_ids = set()

    if (
        not existing.empty
        and "case_id" in existing.columns
    ):

        completed_ids = set(
            existing["case_id"]
            .dropna()
            .astype(str)
        )


    print(
        f"Dataset cases:       {len(merged)}"
    )

    print(
        f"Already evaluated:    {len(completed_ids)}"
    )

    print(
        f"Remaining JEV calls:  "
        f"{len(merged) - len(completed_ids)}"
    )


    # --------------------------------------------------------
    # REMAINING CASES
    # --------------------------------------------------------

    remaining = merged[
        ~merged["case_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()


    # --------------------------------------------------------
    # RUN JEV
    # --------------------------------------------------------

    print("\nStarting JEV controller evaluation...\n")

    new_results = []

    for _, row in tqdm(
        remaining.iterrows(),
        total=len(remaining),
        desc="JEV",
    ):

        case_id = str(
            row["case_id"]
        )

        question = str(
            row["question"]
        )

        stage1_prediction = int(
            row["prediction"]
        )

        stage1_answer = (
            "TRUE"
            if stage1_prediction == 1
            else "FALSE"
        )


        result = call_jev(
            question,
            stage1_answer
        )


        output_row = {

            "case_id":
                case_id,

            "domain":
                row["domain"],

            "difficulty":
                row["difficulty"],

            "question":
                question,

            "label":
                int(row["label"]),

            "stage1_prediction":
                stage1_prediction,

            "stage1_correct":
                int(row["correct"]),

            "stage1_answer":
                stage1_answer,

            "jev_probability_correct":
                result[
                    "jev_probability_correct"
                ],

            "jev_prediction_correct":
                result[
                    "jev_prediction_correct"
                ],

            "latency_seconds":
                result[
                    "latency_seconds"
                ],

            "input_tokens":
                result[
                    "input_tokens"
                ],

            "output_tokens":
                result[
                    "output_tokens"
                ],

            "jev_model":
                result[
                    "jev_model"
                ],

            "error":
                result[
                    "error"
                ],
        }


        new_results.append(
            output_row
        )


        # ----------------------------------------------------
        # SAVE AFTER EACH CALL
        # ----------------------------------------------------

        combined = pd.concat(
            [
                existing,
                pd.DataFrame(
                    new_results
                ),
            ],
            ignore_index=True,
        )


        combined = (
            combined
            .drop_duplicates(
                subset=["case_id"],
                keep="last",
            )
            .sort_values("case_id")
            .reset_index(drop=True)
        )


        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        combined.to_csv(
            OUTPUT_FILE,
            index=False
        )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    final_df = pd.read_csv(
        OUTPUT_FILE
    )


    successful = final_df[
        final_df[
            "jev_probability_correct"
        ].notna()
    ]


    failed = final_df[
        final_df["error"].notna()
        & (
            final_df["error"]
            .astype(str)
            .str.len()
            > 0
        )
    ]


    print("\n" + "=" * 60)
    print("E10 JEV CONTROLLER COMPLETE")
    print("=" * 60)

    print(
        f"Dataset cases:        {len(merged)}"
    )

    print(
        f"JEV results saved:     {len(final_df)}"
    )

    print(
        f"Successful JEV calls:  {len(successful)}"
    )

    print(
        f"Failed JEV calls:      {len(failed)}"
    )


    if len(successful) > 0:

        evaluator_accuracy = (
            successful[
                "jev_prediction_correct"
            ].mean()
        )

        mean_probability = (
            successful[
                "jev_probability_correct"
            ].mean()
        )

        print(
            f"JEV evaluator accuracy: "
            f"{evaluator_accuracy:.4f}"
        )

        print(
            f"Mean P(Stage-1 correct): "
            f"{mean_probability:.4f}"
        )


    if len(failed) > 0:

        print("\nFailed cases:")

        print(
            failed[
                ["case_id", "error"]
            ].to_string(index=False)
        )


    print("\nSaved to:")

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
