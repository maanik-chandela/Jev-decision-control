import csv
import time
import requests

INPUT = "experiments/data/m15_pilot3_v2.csv"
OUTPUT = "experiments/data/m15_pilot3_v2_base_results.csv"
MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """You are a binary reasoning evaluator.

Answer the question with exactly one word:
TRUE or FALSE.

Do not provide an explanation.
"""


def normalize_answer(text):
    text = text.strip().upper()

    if "TRUE" in text and "FALSE" not in text:
        return "TRUE"

    if "FALSE" in text and "TRUE" not in text:
        return "FALSE"

    return "UNKNOWN"


with open(INPUT, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

results = []

for i, row in enumerate(rows, 1):
    prompt = f"""Question:
{row['question']}

Answer only TRUE or FALSE."""

    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    start = time.time()

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        latency = time.time() - start
        response.raise_for_status()

        data = response.json()
        raw = data.get("response", "").strip()
        answer = normalize_answer(raw)

        expected = "TRUE" if row["label"] == "1" else "FALSE"
        correct = answer == expected

        results.append({
            "case_id": row["case_id"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "question": row["question"],
            "label": row["label"],
            "expected_answer": expected,
            "model_answer": answer,
            "correct": int(correct),
            "raw_response": raw,
            "input_tokens": data.get("prompt_eval_count", ""),
            "output_tokens": data.get("eval_count", ""),
            "latency_seconds": round(latency, 4),
            "status": "success"
        })

        print(
            f"[{i:02d}/60] {row['case_id']} "
            f"expected={expected} predicted={answer} "
            f"correct={correct} "
            f"latency={latency:.2f}s"
        )

    except Exception as e:
        latency = time.time() - start

        results.append({
            "case_id": row["case_id"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "question": row["question"],
            "label": row["label"],
            "expected_answer": "",
            "model_answer": "ERROR",
            "correct": 0,
            "raw_response": "",
            "input_tokens": "",
            "output_tokens": "",
            "latency_seconds": round(latency, 4),
            "status": f"error: {e}"
        })

        print(f"[{i:02d}/60] {row['case_id']} ERROR: {e}")


fieldnames = [
    "case_id",
    "domain",
    "difficulty",
    "question",
    "label",
    "expected_answer",
    "model_answer",
    "correct",
    "raw_response",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "status"
]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

successful = sum(r["status"] == "success" for r in results)
correct = sum(r["correct"] == 1 for r in results)

print()
print(f"Saved: {OUTPUT}")
print(f"Rows: {len(results)}")
print(f"Successful: {successful}")
print(f"Errors: {len(results) - successful}")
print(f"Correct: {correct}")
print(f"Accuracy: {correct / len(results):.4f}")
