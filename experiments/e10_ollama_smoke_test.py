import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

tests = [
    {
        "id": "SMOKE_01",
        "question": "Is 17 less than 10?",
        "label": 0,
    },
    {
        "id": "SMOKE_02",
        "question": "A rectangle has length 8 cm and width 3 cm. Is its area 24 square centimeters?",
        "label": 1,
    },
    {
        "id": "SMOKE_03",
        "question": (
            "If all A are B, and all B are C, must all A be C?"
        ),
        "label": 1,
    },
]


def ask_ollama(question):
    prompt = f"""
You are answering a binary research benchmark.

Question:
{question}

Answer with exactly one word:
TRUE
or
FALSE

Do not provide an explanation.
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"].strip()


def parse_answer(text):
    text_upper = text.upper()

    if "TRUE" in text_upper and "FALSE" not in text_upper:
        return 1

    if "FALSE" in text_upper and "TRUE" not in text_upper:
        return 0

    return None


print("=" * 60)
print("E10 OLLAMA SMOKE TEST")
print("=" * 60)

passed = 0

for test in tests:
    print(f"\n{test['id']}")
    print(f"Question: {test['question']}")

    raw = ask_ollama(test["question"])
    prediction = parse_answer(raw)

    print(f"Raw output: {raw}")
    print(f"Parsed prediction: {prediction}")
    print(f"Ground truth: {test['label']}")

    if prediction == test["label"]:
        print("RESULT: PASS")
        passed += 1
    else:
        print("RESULT: FAIL")


print("\n" + "=" * 60)
print(f"PASSED: {passed}/{len(tests)}")
print("=" * 60)
