from pathlib import Path
import csv
import json
import re
import time
import requests

INPUT = Path("experiments/data/m15_pilot.csv")
OUTPUT = Path("experiments/data/m15_base_results.csv")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

SYSTEM_PROMPT = """You are a binary decision model.

Determine whether the given statement is TRUE or FALSE.

Return exactly one final line:
TRUE
or
FALSE

Do not add any other text to the final answer."""

def parse_answer(text):
    matches = re.findall(r"\b(TRUE|FALSE)\b", text.upper())
    if not matches:
        return None
    return matches[-1]

with INPUT.open(newline="", encoding="utf-8") as f:
    cases = list(csv.DictReader(f))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "case_id",
    "domain",
    "difficulty",
    "question",
    "label",
    "model_answer",
    "correct",
    "raw_response",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "status",
]

rows = []

for i, case in enumerate(cases, start=1):
    print(f"[{i}/{len(cases)}] {case['case_id']}")

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Statement:\n{case['question']}",
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    start = time.time()

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300,
        )
        response.raise_for_status()

        data = response.json()
        raw = data["message"]["content"]
        answer = parse_answer(raw)

        label = int(case["label"])
        correct = (
            None if answer is None
            else int((answer == "TRUE") == (label == 1))
        )

        rows.append({
            "case_id": case["case_id"],
            "domain": case["domain"],
            "difficulty": case["difficulty"],
            "question": case["question"],
            "label": label,
            "model_answer": answer,
            "correct": correct,
            "raw_response": raw,
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "latency_seconds": round(time.time() - start, 4),
            "status": "success",
        })

    except Exception as e:
        rows.append({
            "case_id": case["case_id"],
            "domain": case["domain"],
            "difficulty": case["difficulty"],
            "question": case["question"],
            "label": int(case["label"]),
            "model_answer": None,
            "correct": None,
            "raw_response": str(e),
            "input_tokens": None,
            "output_tokens": None,
            "latency_seconds": round(time.time() - start, 4),
            "status": "error",
        })

    time.sleep(0.1)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"Saved: {OUTPUT}")
print(f"Rows: {len(rows)}")
print(f"Successful: {sum(r['status'] == 'success' for r in rows)}")
print(f"Errors: {sum(r['status'] == 'error' for r in rows)}")
