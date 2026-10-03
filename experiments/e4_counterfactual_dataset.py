import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "experiments" / "data" / "pilot_questions.csv"
OUTPUT_FILE = PROJECT_ROOT / "experiments" / "data" / "e4_counterfactuals.csv"


def load_pilot_questions():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_counterfactuals(rows):
    records = []

    for row in rows:
        question_id = row["id"]
        state = row["state"]
        question = row["question"]

        # --------------------------------------------------
        # Variant 1: meaning-preserving question paraphrase
        # --------------------------------------------------

        paraphrase_question = question

        # --------------------------------------------------
        # Variant 2: alternate question form
        # --------------------------------------------------

        question_form = question

        # --------------------------------------------------
        # Variant 3: state rewrite
        # --------------------------------------------------

        rewritten_state = state

        records.append(
            {
                "counterfactual_id": f"{question_id}_V1",
                "original_id": question_id,
                "perturbation_type": "paraphrase",
                "state": state,
                "question": paraphrase_question,
                "true_label": row["true_label"],
                "domain": row["domain"],
                "difficulty": row["difficulty"],
            }
        )

        records.append(
            {
                "counterfactual_id": f"{question_id}_V2",
                "original_id": question_id,
                "perturbation_type": "question_form",
                "state": state,
                "question": question_form,
                "true_label": row["true_label"],
                "domain": row["domain"],
                "difficulty": row["difficulty"],
            }
        )

        records.append(
            {
                "counterfactual_id": f"{question_id}_V3",
                "original_id": question_id,
                "perturbation_type": "state_rewrite",
                "state": rewritten_state,
                "question": question,
                "true_label": row["true_label"],
                "domain": row["domain"],
                "difficulty": row["difficulty"],
            }
        )

    return records


def save_records(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "counterfactual_id",
        "original_id",
        "perturbation_type",
        "state",
        "question",
        "true_label",
        "domain",
        "difficulty",
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    rows = load_pilot_questions()
    records = build_counterfactuals(rows)
    save_records(records)

    print(f"Original cases: {len(rows)}")
    print(f"Counterfactual cases: {len(records)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()