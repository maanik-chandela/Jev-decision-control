import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

E4 = DATA / "e4_counterfactuals_final.csv"
PILOT = DATA / "pilot_results.csv"


df = pd.read_csv(E4)
pilot = pd.read_csv(PILOT)

errors = []


# ---------------------------------------------------------
# Show the actual pilot columns so we never guess incorrectly
# ---------------------------------------------------------

print("Pilot result columns:")
print(list(pilot.columns))
print()


# ---------------------------------------------------------
# Find the original-question identifier column
# ---------------------------------------------------------

possible_id_columns = [
    "question_id",
    "original_id",
    "experiment_id",
    "case_id",
    "counterfactual_id",
    "id",
]

pilot_id_column = None

for column in possible_id_columns:
    if column in pilot.columns:
        pilot_id_column = column
        break

if pilot_id_column is None:
    errors.append(
        "Could not identify the original-question ID column "
        f"in pilot_results.csv. Available columns: {list(pilot.columns)}"
    )


# ---------------------------------------------------------
# 1. Basic structure
# ---------------------------------------------------------

if len(df) != 150:
    errors.append(f"Expected 150 rows, found {len(df)}")

if df["counterfactual_id"].nunique() != 150:
    errors.append("Counterfactual IDs are not unique")

if df["original_id"].nunique() != 50:
    errors.append(
        f"Expected 50 original IDs, found {df['original_id'].nunique()}"
    )


# ---------------------------------------------------------
# 2. Exactly 3 variants per original
# ---------------------------------------------------------

counts = df.groupby("original_id").size()

bad_counts = counts[counts != 3]

if len(bad_counts):
    errors.append(
        f"Originals without exactly 3 variants: "
        f"{bad_counts.to_dict()}"
    )


# ---------------------------------------------------------
# 3. Perturbation balance
# ---------------------------------------------------------

expected_types = {
    "paraphrase": 50,
    "question_form": 50,
    "state_rewrite": 50,
}

actual_types = df["perturbation_type"].value_counts().to_dict()

for kind, expected in expected_types.items():

    actual = actual_types.get(kind, 0)

    if actual != expected:
        errors.append(
            f"{kind}: expected {expected}, found {actual}"
        )


# ---------------------------------------------------------
# 4. Label balance
# ---------------------------------------------------------

labels = df["true_label"].value_counts().to_dict()

if labels.get(1, 0) != 75:
    errors.append(
        f"Expected 75 true cases, found {labels.get(1, 0)}"
    )

if labels.get(0, 0) != 75:
    errors.append(
        f"Expected 75 false cases, found {labels.get(0, 0)}"
    )


# ---------------------------------------------------------
# 5. Every counterfactual has a corresponding original
# ---------------------------------------------------------

if pilot_id_column is not None:

    pilot_ids = set(
        pilot[pilot_id_column]
        .astype(str)
        .str.strip()
    )

    e4_ids = set(
        df["original_id"]
        .astype(str)
        .str.strip()
    )

    missing = sorted(e4_ids - pilot_ids)

    if missing:
        errors.append(
            "Counterfactuals have no matching baseline pilot case: "
            f"{missing}"
        )

    print(
        f"Pilot ID column detected: {pilot_id_column}"
    )

    print(
        f"Baseline IDs available: {len(pilot_ids)}"
    )


# ---------------------------------------------------------
# 6. Expected P001-P050 coverage
# ---------------------------------------------------------

expected_originals = {
    f"P{i:03d}"
    for i in range(1, 51)
}

actual_originals = set(
    df["original_id"]
    .astype(str)
    .str.strip()
)

missing_originals = sorted(
    expected_originals - actual_originals
)

extra_originals = sorted(
    actual_originals - expected_originals
)

if missing_originals:
    errors.append(
        f"Missing original IDs: {missing_originals}"
    )

if extra_originals:
    errors.append(
        f"Unexpected original IDs: {extra_originals}"
    )


# ---------------------------------------------------------
# 7. Check generated-wrapper contamination
# ---------------------------------------------------------

combined = (
    df["state"].fillna("").astype(str)
    + " "
    + df["question"].fillna("").astype(str)
)

bad_phrases = [
    "Please determine whether the following is true:",
    "Would you say the following statement is true:",
    "The information provided indicates that The",
]

for phrase in bad_phrases:

    if combined.str.contains(
        phrase,
        regex=False
    ).any():

        errors.append(
            f"Old generated wrapper remains: {phrase}"
        )


# ---------------------------------------------------------
# 8. Check empty text
# ---------------------------------------------------------

for column in ["state", "question"]:

    empty = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty:
        errors.append(
            f"{column} contains {empty} empty values"
        )


# ---------------------------------------------------------
# 9. P039 check
# ---------------------------------------------------------

p039 = df[
    df["original_id"] == "P039"
]

if len(p039) != 3:

    errors.append(
        "P039 does not have exactly 3 variants."
    )

else:

    p039_questions = (
        p039["question"]
        .astype(str)
        .tolist()
    )

    if not any(
        "could x be less than 3" in q
        for q in p039_questions
    ):
        errors.append(
            "P039 V1 does not contain the expected formulation."
        )

    if not any(
        "if x is greater than 5, then x is less than 3"
        in q.lower()
        for q in p039_questions
    ):
        errors.append(
            "P039 V2 does not contain the expected formulation."
        )


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print()
print("=" * 60)

if errors:

    print("E4 VALIDATION FAILED")
    print("=" * 60)

    for error in errors:
        print(f"ERROR: {error}")

    raise SystemExit(1)


print("E4 VALIDATION PASSED")
print("=" * 60)

print(f"Rows:                   {len(df)}")
print(f"Original cases:         {df['original_id'].nunique()}")
print(
    f"Unique counterfactuals: {df['counterfactual_id'].nunique()}"
)

print()
print("Perturbations:")
print(
    df["perturbation_type"]
    .value_counts()
    .sort_index()
)

print()
print("Labels:")
print(
    df["true_label"]
    .value_counts()
    .sort_index()
)

print()
print("P039:")
print(
    p039[
        [
            "counterfactual_id",
            "perturbation_type",
            "state",
            "question",
            "true_label",
        ]
    ].to_string(index=False)
)

print()
print("READY FOR E4 API RUN.")