import csv
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY is missing")


# ============================================================
# RESEARCH BUDGET
# ============================================================

# Your total available credit is $5.
# We intentionally protect $1 as a reserve.
INTERNAL_BUDGET_USD = 4.00

# Current documented JEV input price:
# $42 per 1 billion input tokens.
COST_PER_BILLION_INPUT_TOKENS = 42.00


def calculate_cost(input_tokens):
    return (
        input_tokens / 1_000_000_000
    ) * COST_PER_BILLION_INPUT_TOKENS


# ============================================================
# DATA FILES
# ============================================================

DATA_DIR = PROJECT_ROOT / "experiments" / "data"

JSONL_FILE = DATA_DIR / "jev_decisions.jsonl"
CSV_FILE = DATA_DIR / "jev_decisions.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# QUESTION
# ============================================================

state = (
    "The package was delivered yesterday according to "
    "the tracking record."
)

question_name = "delivered"

question = {
    "type": "noul",
    "instructions": "Was the package delivered?",
    "criteria": {
        "true": "The package was delivered.",
        "false": "The package was not delivered."
    }
}


# ============================================================
# CHECK CURRENT SPENDING
# ============================================================

current_spend = 0.0

if JSONL_FILE.exists():
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                current_spend += record.get("estimated_cost_usd", 0.0)


print(f"Current estimated research spend: ${current_spend:.8f}")
print(f"Internal research limit: ${INTERNAL_BUDGET_USD:.2f}")


# ============================================================
# ESTIMATE THIS REQUEST BEFORE SENDING
# ============================================================

# Conservative estimate for this small request.
estimated_input_tokens = 500

estimated_request_cost = calculate_cost(
    estimated_input_tokens
)

print(
    f"Estimated maximum cost for this request: "
    f"${estimated_request_cost:.8f}"
)

if current_spend + estimated_request_cost > INTERNAL_BUDGET_USD:
    raise RuntimeError(
        "Budget guard stopped the request. "
        "Internal $4 research limit would be exceeded."
    )


# ============================================================
# SEND REQUEST
# ============================================================

url = "https://api.typesafe.ai/v1/systemone"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "jev-latest",
    "state": state,
    "questions": {
        question_name: question
    }
}


response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=60
)


# ============================================================
# PARSE RESPONSE
# ============================================================

try:
    result = response.json()
except ValueError:
    result = {
        "raw_response": response.text
    }


print("\nHTTP status:", response.status_code)

print("\nJEV response:")
print(json.dumps(result, indent=2))


# ============================================================
# COST + RESEARCH RECORD
# ============================================================

usage = result.get("usage", {})

input_tokens = usage.get("input_tokens", 0)
output_tokens = usage.get("output_tokens", 0)

actual_cost = calculate_cost(input_tokens)

new_total = current_spend + actual_cost

decision_number = 1

if JSONL_FILE.exists():
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        decision_number += sum(
            1 for line in f if line.strip()
        )

decision_id = f"D1-{decision_number:04d}"


answer = (
    result
    .get("answers", {})
    .get(question_name, {})
)

record = {
    "decision_id": decision_id,
    "state": state,
    "question_name": question_name,
    "question_type": "noul",
    "instructions": question["instructions"],
    "criteria_true": question["criteria"]["true"],
    "criteria_false": question["criteria"]["false"],
    "model": result.get("model"),
    "noul": answer.get("noul"),
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "estimated_cost_usd": actual_cost,
    "http_status": response.status_code,
}


# ============================================================
# SAVE JSONL
# ============================================================

with open(JSONL_FILE, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")


# ============================================================
# SAVE CSV
# ============================================================

csv_fields = list(record.keys())

csv_exists = CSV_FILE.exists()

with open(
    CSV_FILE,
    "a",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields
    )

    if not csv_exists:
        writer.writeheader()

    writer.writerow(record)


# ============================================================
# FINAL STATUS
# ============================================================

print("\nSaved research record:")
print(f"Decision ID: {decision_id}")
print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Estimated cost: ${actual_cost:.8f}")
print(f"Estimated total spend: ${new_total:.8f}")

print("\nFiles:")
print(JSONL_FILE)
print(CSV_FILE)