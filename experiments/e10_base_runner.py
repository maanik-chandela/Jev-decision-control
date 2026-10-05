#!/usr/bin/env python3

"""
E10 — Local Stage-1 Baseline Runner

Runs the validated E10 computation-allocation dataset through
Qwen3:8b via a local Ollama server.

Input:
    experiments/data/e10_computation_allocation_dataset.csv

Output:
    experiments/data/e10_base_results.csv
"""

from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "e10_computation_allocation_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "e10_base_results.csv"
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL = "qwen3:8b"

TEMPERATURE = 0

REQUEST_TIMEOUT = 120


SYSTEM_PROMPT = """You are a binary decision model.

Answer the user's question using only the information provided
in the question.

Return exactly one token:
TRUE
or
FALSE

Do not return explanations.
Do not return probabilities.
Do not return any other text.
"""


# ============================================================
# BUILD REQUEST
# ============================================================

def build_payload(question: str) -> dict:

    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
        },
    }


# ============================================================
# PARSE RESPONSE
# ============================================================

def parse_prediction(response_json: dict):

    raw_text = (
        response_json
        .get("message", {})
        .get("content", "")
        .strip()
    )

    normalized = raw_text.upper()

    if normalized == "TRUE":
        return 1, raw_text

    if normalized == "FALSE":
        return 0, raw_text

    # Fallback in case Qwen adds a small amount of text.
    tokens = normalized.split()

    if tokens:

        if tokens[0] == "TRUE":
            return 1, raw_text

        if tokens[0] == "FALSE":
            return 0, raw_text

    return None, raw_text


# ============================================================
# CALL OLLAMA
# ============================================================

def call_ollama(question: str):

    payload = build_payload(question)

    start = time.perf_counter()

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        latency = time.perf_counter() - start

        response.raise_for_status()

        data = response.json()

        prediction, raw_text = parse_prediction(data)

        if prediction is None:

            return (
                None,
                raw_text,
                latency,
                "Could not parse TRUE/FALSE",
            )

        return (
            prediction,
            raw_text,
            latency,
            None,
        )

    except Exception as exc:

        latency = time.perf_counter() - start

        return (
            None,
            "",
            latency,
            repr(exc),
        )


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

def load_existing_results():

    if not OUTPUT_FILE.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(OUTPUT_FILE)

    except Exception:

        print(
            "Warning: existing result file could not be read."
        )

        return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("E10 LOCAL BASELINE — QWEN3:8B")
    print("=" * 60)


    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "case_id",
        "domain",
        "difficulty",
        "question",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:

        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )


    # --------------------------------------------------------
    # VALIDATE DATASET
    # --------------------------------------------------------

    assert len(df) == 150, (
        f"Expected 150 rows, got {len(df)}"
    )

    assert df["case_id"].is_unique, (
        "case_id must be unique"
    )

    assert set(df["label"].unique()) == {0, 1}

    counts = df["label"].value_counts().to_dict()

    assert counts == {
        0: 75,
        1: 75,
    }, f"Unexpected label balance: {counts}"


    print(f"Dataset rows: {len(df)}")


    # --------------------------------------------------------
    # LOAD EXISTING RESULTS
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


    print(f"Already completed: {len(completed_ids)}")

    print(
        f"Remaining: "
        f"{len(df) - len(completed_ids)}"
    )


    # --------------------------------------------------------
    # CHECK OLLAMA
    # --------------------------------------------------------

    print("\nChecking Ollama...")

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        models = response.json().get(
            "models",
            [],
        )

        model_names = {
            model.get("name", "")
            for model in models
        }

        if not any(
            name == MODEL
            or name.startswith(MODEL + ":")
            for name in model_names
        ):

            raise RuntimeError(
                f"{MODEL} was not found in Ollama.\n"
                f"Available models: "
                f"{sorted(model_names)}\n\n"
                f"Run:\n"
                f"ollama pull {MODEL}"
            )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to Ollama.\n\n"
            "Open another Terminal and run:\n\n"
            "    ollama serve\n\n"
            "Then run this experiment again."
        )


    print(f"Ollama OK: {MODEL}")


    # --------------------------------------------------------
    # SELECT REMAINING CASES
    # --------------------------------------------------------

    remaining_df = df[
        ~df["case_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()


    # --------------------------------------------------------
    # RUN BASELINE
    # --------------------------------------------------------

    print("\nRunning local Qwen3:8b baseline...\n")

    new_results = []

    for _, row in tqdm(
        remaining_df.iterrows(),
        total=len(remaining_df),
        desc="Qwen3:8b",
    ):

        case_id = str(row["case_id"])

        question = str(row["question"])

        label = int(row["label"])


        prediction, raw_response, latency, error = (
            call_ollama(question)
        )


        if prediction is not None:

            correct = int(
                prediction == label
            )

        else:

            correct = 0


        result = {

            "case_id": case_id,

            "domain": row["domain"],

            "difficulty": row["difficulty"],

            "question": question,

            "label": label,

            "prediction": prediction,

            "correct": correct,

            "raw_response": raw_response,

            "latency_seconds": latency,

            "error": error,

            "model": MODEL,
        }


        new_results.append(result)


        # ----------------------------------------------------
        # SAVE AFTER EVERY CASE
        # ----------------------------------------------------

        combined = pd.concat(
            [
                existing,
                pd.DataFrame(new_results),
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
            exist_ok=True,
        )


        combined.to_csv(
            OUTPUT_FILE,
            index=False,
        )


    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    final_df = pd.read_csv(
        OUTPUT_FILE
    )


    valid = final_df[
        final_df["prediction"].notna()
    ].copy()


    errors = final_df[
        final_df["error"].notna()
        & (
            final_df["error"]
            .astype(str)
            .str.len()
            > 0
        )
    ]


    if len(valid) > 0:

        accuracy = valid["correct"].mean()

    else:

        accuracy = float("nan")


    print("\n" + "=" * 60)

    print("E10 LOCAL BASELINE COMPLETE")

    print("=" * 60)

    print(
        f"Total dataset cases: {len(df)}"
    )

    print(
        f"Results saved:        {len(final_df)}"
    )

    print(
        f"Successful answers:   {len(valid)}"
    )

    print(
        f"Errors:               {len(errors)}"
    )


    if len(valid) > 0:

        print(
            f"Accuracy:             "
            f"{accuracy:.4f}"
        )


    if len(errors) > 0:

        print("\nFailed cases:")

        print(
            errors[
                ["case_id", "error"]
            ].to_string(index=False)
        )


    print("\nSaved to:")

    print(OUTPUT_FILE)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
