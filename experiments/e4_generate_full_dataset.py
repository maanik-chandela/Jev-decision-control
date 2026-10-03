import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "experiments" / "data"

INPUT_FILE = DATA_DIR / "pilot_questions.csv"
OUTPUT_FILE = DATA_DIR / "e4_counterfactuals_full.csv"


def paraphrase_question(question):
    """
    Controlled paraphrase wrapper.

    Keeps the proposition unchanged while changing the wording
    around the yes/no question.
    """
    if question.startswith("Does "):
        return "Is it correct that " + question[5:-1] + "?"
    if question.startswith("Is "):
        return "Would it be correct to say that " + question[3:-1] + "?"
    if question.startswith("Are "):
        return "Would it be accurate to say that " + question[4:-1] + "?"
    if question.startswith("Can "):
        return "Would it be correct to say that " + question[4:-1] + "?"
    if question.startswith("Did "):
        return "Is it correct that " + question[4:-1] + "?"

    return "Would it be correct to say that " + question.rstrip("?") + "?"


def question_form(question):
    """
    Convert the question into an explicit proposition check.
    """
    q = question.rstrip("?")

    if q.startswith("Does "):
        rest = q[5:]
        return "Would the statement that " + rest + " be true?"

    if q.startswith("Is "):
        rest = q[3:]
        return "Would the statement that " + rest + " be true?"

    if q.startswith("Are "):
        rest = q[4:]
        return "Would the statement that " + rest + " be true?"

    if q.startswith("Can "):
        rest = q[4:]
        return "Would the statement that " + rest + " be true?"

    if q.startswith("Did "):
        rest = q[4:]
        return "Would the statement that " + rest + " be true?"

    return "Is the following statement true: " + q + "?"


def state_rewrite(state):
    """
    Controlled state rewrite.

    Preserves the factual content while changing the presentation.
    """
    state = state.strip()

    if state.endswith("."):
        state = state[:-1]

    return "The information provided states that " + state + "."


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 50:
        raise ValueError(
            f"Expected exactly 50 original questions, found {len(rows)}."
        )

    output_rows = []

    for row in rows:

        original_id = row["id"]

        # V1: question paraphrase
        output_rows.append({
            "counterfactual_id": f"{original_id}_V1",
            "original_id": original_id,
            "perturbation_type": "paraphrase",
            "state": row["state"],
            "question": paraphrase_question(row["question"]),
            "true_label": row["true_label"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
        })

        # V2: question-form transformation
        output_rows.append({
            "counterfactual_id": f"{original_id}_V2",
            "original_id": original_id,
            "perturbation_type": "question_form",
            "state": row["state"],
            "question": question_form(row["question"]),
            "true_label": row["true_label"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
        })

        # V3: state rewrite
        output_rows.append({
            "counterfactual_id": f"{original_id}_V3",
            "original_id": original_id,
            "perturbation_type": "state_rewrite",
            "state": state_rewrite(row["state"]),
            "question": row["question"],
            "true_label": row["true_label"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
        })

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

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 60)
    print("FULL E4 COUNTERFACTUAL DATASET")
    print("=" * 60)

    print(f"Original questions: {len(rows)}")
    print(f"Counterfactual rows: {len(output_rows)}")

    print("\nExpected:")
    print("50 originals")
    print("3 variants per original")
    print("150 counterfactuals")

    print("\nPerturbation counts:")

    counts = {}

    for row in output_rows:
        p = row["perturbation_type"]
        counts[p] = counts.get(p, 0) + 1

    for p, count in counts.items():
        print(f"{p}: {count}")

    ids = [row["counterfactual_id"] for row in output_rows]

    print(f"\nUnique IDs: {len(set(ids))}/{len(ids)}")

    duplicates = len(ids) != len(set(ids))

    print(f"Duplicates: {duplicates}")

    labels = {}

    for row in output_rows:
        label = row["true_label"]
        labels[label] = labels.get(label, 0) + 1

    print("\nLabels:")
    for label, count in sorted(labels.items()):
        print(f"{label}: {count}")

    print("\nFirst 6 generated rows:")

    for row in output_rows[:6]:
        print(
            row["counterfactual_id"],
            "|",
            row["perturbation_type"],
            "|",
            row["question"],
        )

    print("\nLast 6 generated rows:")

    for row in output_rows[-6:]:
        print(
            row["counterfactual_id"],
            "|",
            row["perturbation_type"],
            "|",
            row["question"],
        )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()