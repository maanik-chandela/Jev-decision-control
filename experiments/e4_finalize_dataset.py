from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "e4_counterfactuals_full.csv"
OUTPUT_FILE = DATA_DIR / "e4_counterfactuals_final.csv"


# ---------------------------------------------------------------------
# Manually verified semantic transformations for P010-P050.
#
# P001-P009 are deliberately left completely untouched because they
# were already manually verified in the pilot.
# ---------------------------------------------------------------------

corrections = {

    "P010": {
        "V1": "Does water boil at a lower temperature when atmospheric pressure is reduced?",
        "V2": "Is it true that lowering atmospheric pressure lowers water's boiling temperature?",
        "V3": "When atmospheric pressure falls, water's boiling point falls.",
    },

    "P011": {
        "V1": "Are all four sides of a square equal?",
        "V2": "Is it true that a square has four equal sides?",
        "V3": "All four sides of a square are equal.",
    },

    "P012": {
        "V1": "Does 17 qualify as a prime number?",
        "V2": "Is it true that 17 is prime?",
        "V3": "17 is a prime number.",
    },

    "P013": {
        "V1": "Does differentiating x squared give 2x?",
        "V2": "Is it true that the derivative of x squared is 2x?",
        "V3": "Differentiating x squared gives 2x.",
    },

    "P014": {
        "V1": "Do the interior angles of a Euclidean triangle total 180 degrees?",
        "V2": "Is it true that a Euclidean triangle has interior angles summing to 180 degrees?",
        "V3": "In Euclidean geometry, the interior angles of a triangle total 180 degrees.",
    },

    "P015": {
        "V1": "Is zero an even number?",
        "V2": "Is it true that zero is an even integer?",
        "V3": "Zero belongs to the even integers.",
    },

    "P016": {
        "V1": "Is the principal square root of 144 equal to 12?",
        "V2": "Is it true that the principal square root of 144 is 12?",
        "V3": "The principal square root of 144 is 12.",
    },

    "P017": {
        "V1": "Does every rectangle have four right angles?",
        "V2": "Is it true that every rectangle has four right angles?",
        "V3": "Every rectangle has four right angles.",
    },

    "P018": {
        "V1": "Is heads equally likely as tails on one toss of a fair coin?",
        "V2": "Is it true that the probability of heads on one toss of a fair coin is 0.5?",
        "V3": "A fair coin has a 0.5 probability of landing heads on a single toss.",
    },

    "P019": {
        "V1": "If x is greater than 5, is x also greater than 3?",
        "V2": "Is it true that if x is greater than 5, then x is greater than 3?",
        "V3": "Whenever x is greater than 5, x is also greater than 3.",
    },

    "P020": {
        "V1": "Is each term after the first twice the preceding term?",
        "V2": "Is it true that every term after the first is double the previous term?",
        "V3": "Each term in the sequence 2, 4, 8, 16 is twice the preceding term.",
    },

    "P021": {
        "V1": "Is Paris the capital city of France?",
        "V2": "Is it true that Paris is the capital of France?",
        "V3": "France's capital city is Paris.",
    },

    "P022": {
        "V1": "Does the Pacific Ocean have a greater surface area than the Atlantic Ocean?",
        "V2": "Is it true that the Pacific Ocean is larger than the Atlantic Ocean by surface area?",
        "V3": "By surface area, the Pacific Ocean is larger than the Atlantic Ocean.",
    },

    "P023": {
        "V1": "Does the normal human heart contain four chambers?",
        "V2": "Is it true that the human heart normally has four chambers?",
        "V3": "The normal human heart consists of four chambers.",
    },

    "P024": {
        "V1": "Are there seven days in a standard week?",
        "V2": "Is it true that a standard week contains seven days?",
        "V3": "A standard week consists of seven days.",
    },

    "P025": {
        "V1": "Is India situated in South Asia?",
        "V2": "Is it true that India is located in South Asia?",
        "V3": "India is in South Asia.",
    },

    "P026": {
        "V1": "Is the freezing point of water under standard atmospheric pressure 10 degrees Celsius?",
        "V2": "Is it true that water freezes at 10 degrees Celsius under standard atmospheric pressure?",
        "V3": "At standard atmospheric pressure, water freezes at 0 degrees Celsius.",
    },

    "P027": {
        "V1": "Does Earth have two natural satellites?",
        "V2": "Is it true that Earth has two natural satellites?",
        "V3": "The Earth has one natural satellite, the Moon.",
    },

    "P028": {
        "V1": "Is Ag the chemical symbol for gold?",
        "V2": "Is it true that the chemical symbol for gold is Ag?",
        "V3": "The chemical symbol for gold is Au.",
    },

    "P029": {
        "V1": "Is the speed of light in a vacuum approximately 30,000 kilometers per second?",
        "V2": "Is it true that light travels through a vacuum at approximately 30,000 kilometers per second?",
        "V3": "Light travels through a vacuum at approximately 300,000 kilometers per second.",
    },

    "P030": {
        "V1": "Do humans normally use carbon dioxide instead of oxygen for aerobic respiration?",
        "V2": "Is it true that humans require carbon dioxide instead of oxygen for aerobic respiration?",
        "V3": "Humans need oxygen for normal aerobic respiration.",
    },

    "P031": {
        "V1": "Are exactly three sides of a square equal?",
        "V2": "Is it true that a square has three equal sides?",
        "V3": "All four sides of a square are equal.",
    },

    "P032": {
        "V1": "Does 17 qualify as an even number?",
        "V2": "Is it true that 17 is even?",
        "V3": "The number 17 is prime.",
    },

    "P033": {
        "V1": "Is the derivative of x squared just x?",
        "V2": "Is it true that the derivative of x squared is x?",
        "V3": "The derivative of x squared is 2x.",
    },

    "P034": {
        "V1": "Is the sum of a Euclidean triangle's interior angles 90 degrees?",
        "V2": "Is it true that a Euclidean triangle's interior angles sum to 90 degrees?",
        "V3": "In Euclidean geometry, the interior angles of a triangle total 180 degrees.",
    },

    "P035": {
        "V1": "Does zero belong to the odd integers?",
        "V2": "Is it true that zero is an odd integer?",
        "V3": "Zero belongs to the even integers.",
    },

    "P036": {
        "V1": "Does the principal square root of 144 equal 14?",
        "V2": "Is it true that the principal square root of 144 is 14?",
        "V3": "The principal square root of 144 is 12.",
    },

    "P037": {
        "V1": "Is it possible for a rectangle to contain five right angles?",
        "V2": "Is it true that a rectangle can have five right angles?",
        "V3": "Every rectangle has four right angles.",
    },

    "P038": {
        "V1": "Can a fair coin have a 0.9 probability of heads on one toss?",
        "V2": "Is it true that the probability of heads on one toss of a fair coin is 0.9?",
        "V3": "A fair coin has a 0.5 probability of landing heads on a single toss.",
    },

    "P039": {
        "V1": "If x is greater than 5, could x be less than 3?",
        "V2": "Is it true that if x is greater than 5, then x is less than 3?",
        "V3": "Whenever x is greater than 5, x is also greater than 3.",
    },

    "P040": {
        "V1": "Is each term after the first one greater than the previous term by 1?",
        "V2": "Is it true that each term after the first equals the previous term plus 1?",
        "V3": "Each term in the sequence 2, 4, 8, 16 is twice the preceding term.",
    },

    "P041": {
        "V1": "Is Berlin the capital city of France?",
        "V2": "Is it true that Berlin is the capital of France?",
        "V3": "France's capital city is Paris.",
    },

    "P042": {
        "V1": "Does the Atlantic Ocean have a larger surface area than the Pacific Ocean?",
        "V2": "Is it true that the Atlantic Ocean is larger than the Pacific Ocean by surface area?",
        "V3": "By surface area, the Pacific Ocean is larger than the Atlantic Ocean.",
    },

    "P043": {
        "V1": "Does the normal human heart contain two chambers?",
        "V2": "Is it true that the human heart normally has two chambers?",
        "V3": "The normal human heart consists of four chambers.",
    },

    "P044": {
        "V1": "Are there ten days in a standard week?",
        "V2": "Is it true that a standard week contains ten days?",
        "V3": "A standard week consists of seven days.",
    },

    "P045": {
        "V1": "Is India situated in South America?",
        "V2": "Is it true that India is located in South America?",
        "V3": "India is in South Asia.",
    },

    "P046": {
        "V1": "Did the student achieve exactly 90 percent on the test?",
        "V2": "Is it true that the student scored exactly 90 percent on the test?",
        "V3": "A student answered 8 of 10 test questions correctly.",
    },

    "P047": {
        "V1": "Do the blue balls outnumber the red balls?",
        "V2": "Is it true that there are more blue balls than red balls?",
        "V3": "The box contains three red balls and two blue balls.",
    },

    "P048": {
        "V1": "Was the train's journey exactly three hours long?",
        "V2": "Is it true that the train traveled for exactly three hours?",
        "V3": "The train departs at 10:00 and arrives at 12:00.",
    },

    "P049": {
        "V1": "Are 120 books still left in the library?",
        "V2": "Is it true that the library has 120 books remaining?",
        "V3": "The library initially has 120 books and removes 20 of them.",
    },

    "P050": {
        "V1": "Are more than 40 units of the item still left?",
        "V2": "Is it true that the shop has more than 40 units remaining?",
        "V3": "The shop starts with 50 units of the item and sells 15 units.",
    },
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "counterfactual_id",
        "original_id",
        "perturbation_type",
        "state",
        "question",
        "true_label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # -------------------------------------------------------------
    # Preserve P001-P009 exactly.
    # Modify only P010-P050.
    # -------------------------------------------------------------

    for original_id, variants in corrections.items():

        mask = df["original_id"] == original_id

        rows = df.loc[mask]

        if len(rows) != 3:
            raise ValueError(
                f"{original_id} has {len(rows)} rows; expected exactly 3."
            )

        for idx in rows.index:

            variant_id = str(df.at[idx, "counterfactual_id"])

            if variant_id.endswith("_V1"):
                df.at[idx, "question"] = variants["V1"]

            elif variant_id.endswith("_V2"):
                df.at[idx, "question"] = variants["V2"]

            elif variant_id.endswith("_V3"):
                df.at[idx, "state"] = variants["V3"]

            else:
                raise ValueError(
                    f"Unexpected counterfactual ID: {variant_id}"
                )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    errors = []

    if len(df) != 150:
        errors.append(f"Expected 150 rows, found {len(df)}.")

    if df["counterfactual_id"].duplicated().any():
        errors.append("Duplicate counterfactual IDs detected.")

    original_counts = df.groupby("original_id").size()

    if not (original_counts == 3).all():
        errors.append(
            "Not every original question has exactly 3 counterfactuals."
        )

    perturbation_counts = df["perturbation_type"].value_counts()

    for perturbation in ["paraphrase", "question_form", "state_rewrite"]:
        count = perturbation_counts.get(perturbation, 0)

        if count != 50:
            errors.append(
                f"{perturbation}: expected 50 rows, found {count}."
            )

    label_counts = df["true_label"].value_counts()

    if label_counts.get(1, 0) != 75:
        errors.append(
            f"True labels: expected 75, found {label_counts.get(1, 0)}."
        )

    if label_counts.get(0, 0) != 75:
        errors.append(
            f"False labels: expected 75, found {label_counts.get(0, 0)}."
        )

    # Make sure all expected IDs exist.
    expected_ids = {
        f"P{i:03d}_V{v}"
        for i in range(1, 51)
        for v in range(1, 4)
    }

    actual_ids = set(df["counterfactual_id"])

    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids

    if missing_ids:
        errors.append(
            f"Missing IDs: {sorted(missing_ids)}"
        )

    if extra_ids:
        errors.append(
            f"Unexpected IDs: {sorted(extra_ids)}"
        )

    # Check for the exact problematic generic wrappers from the old
    # generated dataset.
    bad_phrases = [
        "Please determine whether the following is true:",
        "Would you say the following statement is true:",
        "The information provided indicates that The",
    ]

    text_columns = (
        df["state"].fillna("").astype(str)
        + " "
        + df["question"].fillna("").astype(str)
    )

    for phrase in bad_phrases:
        if text_columns.str.contains(phrase, regex=False).any():
            errors.append(
                f"Old generated wrapper still present: {phrase}"
            )

    if errors:
        print("\nDATASET VALIDATION FAILED\n")

        for error in errors:
            print(f"ERROR: {error}")

        raise SystemExit(1)

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("FINAL E4 DATASET CREATED")
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"Original questions: {df['original_id'].nunique()}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nPerturbation counts:")
    print(df["perturbation_type"].value_counts().sort_index())

    print("\nLabels:")
    print(df["true_label"].value_counts().sort_index())

    print("\nUnique IDs:")
    print(df["counterfactual_id"].nunique())

    print("\nDuplicates:")
    print(df["counterfactual_id"].duplicated().any())

    print("\nP001-P009 preserved:")
    print("YES — no rows from P001-P009 were modified.")

    print("\nSample corrected rows:")
    preview = df[
        df["original_id"].isin(["P010", "P026", "P039", "P046", "P050"])
    ][
        [
            "counterfactual_id",
            "original_id",
            "perturbation_type",
            "state",
            "question",
            "true_label",
        ]
    ]

    print(preview.to_string(index=False))

    print("\nDataset is ready for manual semantic review.")
    print("DO NOT RUN JEV YET.")


if __name__ == "__main__":
    main()