from pathlib import Path
import ast
import csv
import os
import time
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "experiments" / "data"

SOURCE_FILE = PROJECT_ROOT / "experiments" / "e5_create_dataset.py"
OUTPUT_FILE = DATA_DIR / "e5_original_results.csv"

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# API CONFIG
# ============================================================

API_URL = "https://api.typesafe.ai/v1/systemone"

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "TYPESAFE_API_KEY was not found in .env"
    )

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# ============================================================
# EXTRACT FROZEN CASES DIRECTLY FROM SOURCE
# ============================================================

def load_cases_from_source():
    """
    Extract the literal CASES list from e5_create_dataset.py.

    This avoids manually reconstructing the 60 originals.
    No transformation functions are executed.
    """

    source = SOURCE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CASES":
                    cases = ast.literal_eval(node.value)

                    if not isinstance(cases, list):
                        raise RuntimeError("CASES is not a list.")

                    return cases

    raise RuntimeError("Could not find CASES in source file.")


CASES = load_cases_from_source()


# ============================================================
# VALIDATE FROZEN ORIGINAL DATASET
# ============================================================

if len(CASES) != 60:
    raise RuntimeError(
        f"Expected 60 original cases, found {len(CASES)}"
    )

labels = [case["label"] for case in CASES]

if labels.count(1) != 30:
    raise RuntimeError(
        f"Expected 30 true cases, found {labels.count(1)}"
    )

if labels.count(0) != 30:
    raise RuntimeError(
        f"Expected 30 false cases, found {labels.count(0)}"
    )

domains = {}
for case in CASES:
    domains.setdefault(case["domain"], 0)
    domains[case["domain"]] += 1

expected_domains = {
    "mathematics": 12,
    "probability": 12,
    "formal_logic": 12,
    "science_reasoning": 12,
    "data_reasoning": 12,
}

if domains != expected_domains:
    raise RuntimeError(
        f"Unexpected domain distribution: {domains}"
    )

print("=" * 60)
print("E5 ORIGINAL DATASET VALIDATION")
print("=" * 60)
print(f"Original cases: {len(CASES)}")
print(f"True labels:    {labels.count(1)}")
print(f"False labels:   {labels.count(0)}")
print(f"Domains:        {domains}")
print("Dataset source: e5_create_dataset.py")
print("Status:         VALIDATED")
print("=" * 60)


# ============================================================
# JEV CALL
# ============================================================

def call_jev(state, question):
    payload = {
        "state": state,
        "model": "jev-latest",
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": question,
            }
        },
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        print("HTTP ERROR:", response.status_code)
        print("RESPONSE:", response.text)
        response.raise_for_status()

    data = response.json()

    answer = data["answers"]["q1"]

    probability = float(answer["noul"])

    prediction = 1 if probability >= 0.5 else 0

    usage = data.get("usage", {})

    return {
        "probability": probability,
        "prediction": prediction,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


# ============================================================
# RUN ORIGINALS
# ============================================================

results = []

print()
print("Running JEV on 60 ORIGINAL cases...")
print("No adversarial variants are being evaluated in this run.")
print()

for case in tqdm(CASES, desc="E5 originals"):

    try:
        result = call_jev(
            state=case["state"],
            question=case["question"],
        )

        probability = result["probability"]
        prediction = result["prediction"]
        label = case["label"]

        results.append({
            "original_id": case["id"],
            "domain": case["domain"],
            "state": case["state"],
            "question": case["question"],
            "label": label,
            "probability": probability,
            "prediction": prediction,
            "correct": int(prediction == label),
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        })

    except Exception as e:

        print()
        print(f"FAILED: {case['id']}")
        print(f"Error: {e}")

        results.append({
            "original_id": case["id"],
            "domain": case["domain"],
            "state": case["state"],
            "question": case["question"],
            "label": case["label"],
            "probability": "",
            "prediction": "",
            "correct": "",
            "input_tokens": "",
            "output_tokens": "",
        })

    # Small delay to avoid unnecessary burst traffic.
    time.sleep(0.15)


# ============================================================
# SAVE
# ============================================================

fieldnames = [
    "original_id",
    "domain",
    "state",
    "question",
    "label",
    "probability",
    "prediction",
    "correct",
    "input_tokens",
    "output_tokens",
]

with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

successful = [
    r for r in results
    if r["probability"] != ""
]

correct = [
    r for r in successful
    if r["correct"] == 1
]

total_input = sum(
    int(r["input_tokens"])
    for r in successful
)

total_output = sum(
    int(r["output_tokens"])
    for r in successful
)

# TypeSafe JEV input price:
# $42 / billion input tokens
estimated_cost = total_input * 42 / 1_000_000_000

print()
print("=" * 60)
print("E5 ORIGINAL EVALUATION COMPLETE")
print("=" * 60)

print(f"Successful:        {len(successful)}/60")
print(f"Failed:            {60 - len(successful)}")

if successful:
    print(
        f"Accuracy:          "
        f"{len(correct) / len(successful):.4f}"
    )

print(f"Input tokens:      {total_input:,}")
print(f"Output tokens:     {total_output:,}")
print(f"Estimated cost:    ${estimated_cost:.8f}")

print()
print(f"Output:")
print(OUTPUT_FILE)

print("=" * 60)
