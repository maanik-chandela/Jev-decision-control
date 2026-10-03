import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "experiments" / "data"

ORIGINAL_FILE = DATA_DIR / "pilot_questions.csv"
EXISTING_E4_FILE = DATA_DIR / "e4_counterfactuals.csv"
OUTPUT_FILE = DATA_DIR / "e4_counterfactuals_full.csv"


def state_rewrite(state):
    """
    Controlled rewrite that preserves the original proposition.
    """
    state = state.strip()

    if state.endswith("."):
        state = state[:-1]

    return f"The information provided indicates that {state}."


def question_wrapper(question):
    """
    Controlled surface transformation.

    IMPORTANT:
    We deliberately preserve the original question rather than
    attempting unreliable subject/verb reconstruction.
    """
    question = question.strip()

    return f"Please determine whether the following is true: {question}"


def question_form_wrapper(question):
    """
    Alternative question presentation that preserves the original
    proposition verbatim.
    """
    question = question.strip()

    return f"Would you say the following statement is true: {question}"


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():

    originals = load_csv(ORIGINAL_FILE)
    existing = load_csv(EXISTING_E4_FILE)

    if len(originals) != 50:
        raise ValueError(
            f"Expected 50 original questions, found {len(originals)}"
        )

    if len(existing) != 27:
        raise ValueError(
            f"Expected the existing validated E4 pilot to contain "
            f"27 rows, found {len(existing)}"
        )

    # ---------------------------------------------------------
    # Preserve the existing manually verified 27 rows EXACTLY.
    # ---------------------------------------------------------

    existing_ids = {
        row["counterfactual_id"]
        for row in existing
    }

    expected_existing_ids = {
        f"P{i:03d}_V{v}"
        for i in range(1, 10)
        for v in range(1, 4)
    }

    if existing_ids != expected_existing_ids:
        missing = expected_existing_ids - existing_ids
        extra = existing_ids - expected_existing_ids

        raise ValueError(
            f"Existing E4 pilot IDs are incorrect.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )

    output_rows = list(existing)

    # ---------------------------------------------------------
    # Generate candidates ONLY for P010-P050.
    # ---------------------------------------------------------

    for row in originals:

        original_id = row["id"]

        number = int(original_id[1:])

        if number <= 9:
            continue

        # V1
        output_rows.append({
            "counterfactual_id": f"{original_id}_V1",
            "original_id": original_id,
            "perturbation_type": "paraphrase",
            "state": row["state"],
            "question": question_wrapper(row["question"]),
            "true_label": row["true_label"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
        })

        # V2
        output_rows.append({
            "counterfactual_id": f"{original_id}_V2",
            "original_id": original_id,
            "perturbation_type": "question_form",
            "state": row["state"],
            "question": question_form_wrapper(row["question"]),
            "true_label": row["true_label"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
        })

        # V3
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

    # ---------------------------------------------------------
    # Sort deterministically.
    # ---------------------------------------------------------

    output_rows.sort(
        key=lambda x: (
            int(x["original_id"][1:]),
            int(x["counterfactual_id"].split("_V")[1])
        )
    )

    # ---------------------------------------------------------
    # Validation.
    # ---------------------------------------------------------

    if len(output_rows) != 150:
        raise ValueError(
            f"Expected 150 rows, found {len(output_rows)}"
        )

    ids = [
        row["counterfactual_id"]
        for row in output_rows
    ]

    if len(set(ids)) != 150:
        raise ValueError("Duplicate counterfactual IDs detected.")

    original_ids = {
        row["original_id"]
        for row in output_rows
    }

    if len(original_ids) != 50:
        raise ValueError(
            f"Expected 50 original IDs, found {len(original_ids)}"
        )

    for original_id in sorted(original_ids):
        variants = [
            row for row in output_rows
            if row["original_id"] == original_id
        ]

        if len(variants) != 3:
            raise ValueError(
                f"{original_id} does not have exactly 3 variants."
            )

    # Check labels.
    labels = [int(row["true_label"]) for row in output_rows]

    if labels.count(0) != 75:
        raise ValueError(
            f"Expected 75 false labels, found {labels.count(0)}"
        )

    if labels.count(1) != 75:
        raise ValueError(
            f"Expected 75 true labels, found {labels.count(1)}"
        )

    # ---------------------------------------------------------
    # Write.
    # ---------------------------------------------------------

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
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 60)
    print("CORRECTED E4 DATASET")
    print("=" * 60)

    print(f"Original questions: {len(originals)}")
    print(f"Counterfactual rows: {len(output_rows)}")

    print()
    print("Existing verified rows preserved:", len(existing))

    print()
    print("Perturbation counts:")

    for perturbation in [
        "paraphrase",
        "question_form",
        "state_rewrite"
    ]:
        count = sum(
            row["perturbation_type"] == perturbation
            for row in output_rows
        )
        print(f"{perturbation}: {count}")

    print()
    print("Unique IDs:", len(set(ids)))
    print("Duplicates:", len(set(ids)) != len(ids))

    print()
    print("Labels:")
    print("0:", labels.count(0))
    print("1:", labels.count(1))

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()