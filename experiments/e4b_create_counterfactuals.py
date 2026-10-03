import csv
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "experiments" / "data"

INPUT = DATA / "e4b_hard_baseline_results.csv"
OUTPUT = DATA / "e4b_counterfactuals.csv"


# Selected moderate/low-confidence cases.
# H015 is deliberately excluded because its "necessarily 64"
# formulation introduces unnecessary inductive ambiguity.

selected = {
    "H005",
    "H031",
    "H034",
    "H035",
    "H039",
    "P031",

    # High-confidence controls
    "H002",
    "H003",
    "H007",
    "H009",
    "H013",
    "H021",
}


def transform_paraphrase(state, question):
    return state, question


def make_variants(candidate_id, state, question, label):
    variants = []

    # ---------------------------------------------------------
    # Variant 1: Paraphrase
    # ---------------------------------------------------------

    paraphrases = {
        "H002": (
            "A value x satisfies 3x - 4 = 11.",
            "Is x greater than 4?",
        ),
        "H003": (
            "Consider a rectangle measuring 8 cm by 5 cm.",
            "Is its perimeter more than 20 cm?",
        ),
        "H005": (
            "A bag has 3 red balls and 7 blue balls, and one ball is chosen uniformly.",
            "Is the probability of choosing a red ball above 0.25?",
        ),
        "H007": (
            "One roll is made with a fair six-sided die.",
            "Is the probability of getting an even result greater than 0.4?",
        ),
        "H009": (
            "Five numbers have an arithmetic mean of 12.",
            "Do the five numbers have a total sum of exactly 60?",
        ),
        "H013": (
            "A triangle has side lengths of 3 cm, 4 cm, and 5 cm.",
            "Does this triangle have a right angle?",
        ),
        "H021": (
            "Whenever this machine overheats, its alarm activates. The machine overheated.",
            "Did the alarm activate?",
        ),
        "H031": (
            "Someone travels from Delhi to Mumbai.",
            "Is it necessary for the entire trip to be westward?",
        ),
        "H034": (
            "A person is more than 20 years old.",
            "Is that person necessarily at least 30 years old?",
        ),
        "H035": (
            "A box contains ten or more objects.",
            "Does that mean the box contains exactly ten objects?",
        ),
        "H039": (
            "A store starts with 80 items, sells 31, and then receives 10 more.",
            "Does it end up with more than 60 items?",
        ),
        "P031": (
            "If x is greater than 5, then x is greater than 3.",
            "Could x nevertheless be less than 3?",
        ),
    }

    s1, q1 = paraphrases[candidate_id]

    variants.append(
        {
            "variant_id": f"{candidate_id}_V1",
            "perturbation_type": "paraphrase",
            "state": s1,
            "question": q1,
            "true_label": label,
        }
    )

    # ---------------------------------------------------------
    # Variant 2: Question form
    # ---------------------------------------------------------

    question_forms = {
        "H002": "Is it true that x is greater than 4?",
        "H003": "Would the perimeter of the rectangle exceed 20 cm?",
        "H005": "Could the probability of selecting a red ball be greater than 0.25?",
        "H007": "Is an even outcome more probable than 0.4?",
        "H009": "Must the sum of the five numbers be 60?",
        "H013": "Is the triangle right-angled?",
        "H021": "Would the alarm activate?",
        "H031": "Must the journey be entirely westward?",
        "H034": "Must the person's age be at least 30?",
        "H035": "Must the box contain exactly 10 objects?",
        "H039": "Will the store have more than 60 items afterward?",
        "P031": "Does x > 5 imply x < 3?",
    }

    variants.append(
        {
            "variant_id": f"{candidate_id}_V2",
            "perturbation_type": "question_form",
            "state": state,
            "question": question_forms[candidate_id],
            "true_label": label,
        }
    )

    # ---------------------------------------------------------
    # Variant 3: State rewrite
    # ---------------------------------------------------------

    state_rewrites = {
        "H002": "For a number x, subtracting 4 from three times x gives 11.",
        "H003": "The rectangle's dimensions are 8 cm and 5 cm.",
        "H005": "Among 10 equally likely balls, 3 are red and 7 are blue.",
        "H007": "A standard fair die with six faces is rolled once.",
        "H009": "Five values together average to 12.",
        "H013": "The three sides of a triangle measure 3 cm, 4 cm, and 5 cm.",
        "H021": "The machine is known to have overheated, and overheating activates its alarm.",
        "H031": "The destination is Mumbai and the starting point is Delhi.",
        "H034": "The person's age is greater than 20 years.",
        "H035": "There are at least ten objects inside the box.",
        "H039": "Initially there are 80 items; 31 are sold and 10 are subsequently added.",
        "P031": "Whenever x exceeds 5, x also exceeds 3.",
    }

    variants.append(
        {
            "variant_id": f"{candidate_id}_V3",
            "perturbation_type": "state_rewrite",
            "state": state_rewrites[candidate_id],
            "question": question,
            "true_label": label,
        }
    )

    return variants


with INPUT.open(encoding="utf-8") as f:
    baseline_rows = list(csv.DictReader(f))


baseline_lookup = {
    row["candidate_id"]: row
    for row in baseline_rows
}


missing = selected - set(baseline_lookup)

# P031 comes from the original E4 baseline rather than the
# E4b hard-screening file.
if "P031" in missing:
    p031_file = DATA / "pilot_results.csv"

    if not p031_file.exists():
        raise FileNotFoundError(
            "P031 is not available. Expected experiments/data/pilot_results.csv"
        )

    with p031_file.open(encoding="utf-8") as f:
        pilot_rows = list(csv.DictReader(f))

    p031_rows = [
        row for row in pilot_rows
        if row.get("question_id") == "P031"
    ]

    if not p031_rows:
        raise RuntimeError("Could not find P031 in pilot_results.csv")

    row = p031_rows[0]

    baseline_lookup["P031"] = {
        "candidate_id": "P031",
        "state": row["state"],
        "question": row["question"],
        "true_label": row["true_label"],
        "jev_probability": row["jev_probability"],
    }


missing = selected - set(baseline_lookup)

if missing:
    raise RuntimeError(
        f"Missing selected candidates: {sorted(missing)}"
    )


output_rows = []

for candidate_id in sorted(selected):

    row = baseline_lookup[candidate_id]

    variants = make_variants(
        candidate_id,
        row["state"],
        row["question"],
        int(row["true_label"]),
    )

    for variant in variants:
        output_rows.append(
            {
                "variant_id": variant["variant_id"],
                "candidate_id": candidate_id,
                "baseline_probability": float(
                    row["jev_probability"]
                ),
                "domain": row.get("domain", "unknown"),
                "perturbation_type": variant[
                    "perturbation_type"
                ],
                "state": variant["state"],
                "question": variant["question"],
                "true_label": variant["true_label"],
            }
        )


fields = [
    "variant_id",
    "candidate_id",
    "baseline_probability",
    "domain",
    "perturbation_type",
    "state",
    "question",
    "true_label",
]


with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(output_rows)


print("=" * 60)
print("E4b COUNTERFACTUAL DATASET CREATED")
print("=" * 60)

print(f"Selected originals: {len(selected)}")
print(f"Counterfactual rows: {len(output_rows)}")
print()

print("Selected candidates:")

for candidate_id in sorted(selected):
    row = baseline_lookup[candidate_id]

    print(
        f"  {candidate_id}: "
        f"baseline p={float(row['jev_probability']):.2f}"
    )

print()
print("Perturbation counts:")

from collections import Counter

counts = Counter(
    row["perturbation_type"]
    for row in output_rows
)

for k, v in counts.items():
    print(f"  {k}: {v}")

print()
print(f"Output: {OUTPUT}")
