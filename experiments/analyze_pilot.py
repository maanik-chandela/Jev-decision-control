import math
from pathlib import Path

import pandas as pd


# ========================================================
# PATHS
# ========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "experiments" / "data"

INPUT_FILE = DATA_DIR / "pilot_results.csv"

OUTPUT_FILE = DATA_DIR / "pilot_analysis.csv"


# ========================================================
# CONFIGURATION
# ========================================================

PROBABILITY_COLUMN = "jev_probability"
LABEL_COLUMN = "true_label"
PREDICTION_COLUMN = "predicted_label"


# ========================================================
# LOAD DATA
# ========================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find pilot results:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)


if df.empty:
    raise ValueError("pilot_results.csv is empty.")


required_columns = [
    PROBABILITY_COLUMN,
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    "domain",
    "difficulty",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ========================================================
# BASIC DATA VALIDATION
# ========================================================

df[PROBABILITY_COLUMN] = pd.to_numeric(
    df[PROBABILITY_COLUMN]
)

df[LABEL_COLUMN] = pd.to_numeric(
    df[LABEL_COLUMN]
).astype(int)

df[PREDICTION_COLUMN] = pd.to_numeric(
    df[PREDICTION_COLUMN]
).astype(int)


if not df[PROBABILITY_COLUMN].between(
    0, 1
).all():

    raise ValueError(
        "Some JEV probabilities are outside [0, 1]."
    )


if not df[LABEL_COLUMN].isin(
    [0, 1]
).all():

    raise ValueError(
        "true_label must contain only 0 or 1."
    )


if not df[PREDICTION_COLUMN].isin(
    [0, 1]
).all():

    raise ValueError(
        "predicted_label must contain only 0 or 1."
    )


# ========================================================
# BASIC COUNTS
# ========================================================

n = len(df)

positive_cases = int(
    (df[LABEL_COLUMN] == 1).sum()
)

negative_cases = int(
    (df[LABEL_COLUMN] == 0).sum()
)

correct = int(
    (
        df[PREDICTION_COLUMN]
        == df[LABEL_COLUMN]
    ).sum()
)

incorrect = n - correct

accuracy = correct / n


# ========================================================
# CONFUSION MATRIX
# ========================================================

true_positive = int(
    (
        (df[LABEL_COLUMN] == 1)
        & (df[PREDICTION_COLUMN] == 1)
    ).sum()
)

true_negative = int(
    (
        (df[LABEL_COLUMN] == 0)
        & (df[PREDICTION_COLUMN] == 0)
    ).sum()
)

false_positive = int(
    (
        (df[LABEL_COLUMN] == 0)
        & (df[PREDICTION_COLUMN] == 1)
    ).sum()
)

false_negative = int(
    (
        (df[LABEL_COLUMN] == 1)
        & (df[PREDICTION_COLUMN] == 0)
    ).sum()
)


# ========================================================
# CLASSIFICATION METRICS
# ========================================================

precision = (
    true_positive
    / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0.0
)

recall = (
    true_positive
    / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0.0
)

specificity = (
    true_negative
    / (true_negative + false_positive)
    if (true_negative + false_positive) > 0
    else 0.0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)


# ========================================================
# PROBABILITY METRICS
# ========================================================

p = df[PROBABILITY_COLUMN]
y = df[LABEL_COLUMN]


# Brier score
brier_score = (
    ((p - y) ** 2).mean()
)


# Mean absolute probability error
mean_absolute_probability_error = (
    (p - y).abs().mean()
)


# ========================================================
# LOG LOSS
# ========================================================

# Avoid log(0) by clipping probabilities.
epsilon = 1e-15

p_clipped = p.clip(
    lower=epsilon,
    upper=1 - epsilon,
)


log_loss = -(
    y * p_clipped.apply(math.log)
    +
    (1 - y)
    * (1 - p_clipped).apply(math.log)
).mean()


# ========================================================
# CONFIDENCE
# ========================================================

# For a binary decision, confidence in the predicted
# class is:
#
# predicted true  -> p
# predicted false -> 1 - p

df["prediction_confidence"] = (
    df[PREDICTION_COLUMN]
    * p
    +
    (1 - df[PREDICTION_COLUMN])
    * (1 - p)
)


mean_confidence = (
    df["prediction_confidence"].mean()
)


min_confidence = (
    df["prediction_confidence"].min()
)

max_confidence = (
    df["prediction_confidence"].max()
)


# ========================================================
# TRUE-CASE PROBABILITY STATISTICS
# ========================================================

true_df = df[
    df[LABEL_COLUMN] == 1
]

false_df = df[
    df[LABEL_COLUMN] == 0
]


true_probability_mean = (
    true_df[PROBABILITY_COLUMN].mean()
)

true_probability_median = (
    true_df[PROBABILITY_COLUMN].median()
)

true_probability_min = (
    true_df[PROBABILITY_COLUMN].min()
)

true_probability_max = (
    true_df[PROBABILITY_COLUMN].max()
)


# ========================================================
# FALSE-CASE PROBABILITY STATISTICS
# ========================================================

false_probability_mean = (
    false_df[PROBABILITY_COLUMN].mean()
)

false_probability_median = (
    false_df[PROBABILITY_COLUMN].median()
)

false_probability_min = (
    false_df[PROBABILITY_COLUMN].min()
)

false_probability_max = (
    false_df[PROBABILITY_COLUMN].max()
)


# ========================================================
# PROBABILITY SEPARATION
# ========================================================

probability_separation = (
    true_probability_min
    - false_probability_max
)


# ========================================================
# ECE
# ========================================================

def calculate_ece(dataframe, number_of_bins=10):
    """
    Calculate Expected Calibration Error.

    Bins are equally spaced between 0 and 1.
    """

    probabilities = dataframe[
        PROBABILITY_COLUMN
    ].to_numpy()

    labels = dataframe[
        LABEL_COLUMN
    ].to_numpy()

    bin_edges = [
        i / number_of_bins
        for i in range(number_of_bins + 1)
    ]

    ece = 0.0

    for i in range(number_of_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == number_of_bins - 1:

            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not mask.any():
            continue

        bin_probabilities = probabilities[
            mask
        ]

        bin_labels = labels[
            mask
        ]

        mean_confidence = (
            bin_probabilities.mean()
        )

        observed_accuracy = (
            bin_labels.mean()
        )

        bin_fraction = (
            mask.sum()
            / len(probabilities)
        )

        ece += (
            bin_fraction
            * abs(
                mean_confidence
                - observed_accuracy
            )
        )

    return ece


ece_5 = calculate_ece(
    df,
    number_of_bins=5,
)

ece_10 = calculate_ece(
    df,
    number_of_bins=10,
)


# ========================================================
# CONFIDENTLY WRONG CASES
# ========================================================

# These are particularly important for our research.
# A threshold of 0.90 means:
#
# predicted true with >= 0.90
# OR
# predicted false with <= 0.10

confident_wrong = df[
    (
        (
            (y == 1)
            & (p < 0.10)
        )
        |
        (
            (y == 0)
            & (p > 0.90)
        )
    )
]


# ========================================================
# UNCERTAIN CASES
# ========================================================

uncertain_cases = df[
    p.between(
        0.40,
        0.60,
    )
]


# ========================================================
# DOMAIN ANALYSIS
# ========================================================

domain_results = (
    df.groupby("domain")
    .agg(
        cases=("domain", "size"),
        accuracy=(
            PREDICTION_COLUMN,
            lambda x: (
                x
                == df.loc[
                    x.index,
                    LABEL_COLUMN,
                ]
            ).mean(),
        ),
        mean_probability=(
            PROBABILITY_COLUMN,
            "mean",
        ),
        mean_confidence=(
            "prediction_confidence",
            "mean",
        ),
        brier_score=(
            PROBABILITY_COLUMN,
            lambda x: (
                (
                    x
                    - df.loc[
                        x.index,
                        LABEL_COLUMN,
                    ]
                )
                ** 2
            ).mean(),
        ),
    )
    .reset_index()
)


# ========================================================
# DIFFICULTY ANALYSIS
# ========================================================

difficulty_results = (
    df.groupby("difficulty")
    .agg(
        cases=("difficulty", "size"),
        accuracy=(
            PREDICTION_COLUMN,
            lambda x: (
                x
                == df.loc[
                    x.index,
                    LABEL_COLUMN,
                ]
            ).mean(),
        ),
        mean_probability=(
            PROBABILITY_COLUMN,
            "mean",
        ),
        mean_confidence=(
            "prediction_confidence",
            "mean",
        ),
    )
    .reset_index()
)


# ========================================================
# SAVE ANALYSIS SUMMARY
# ========================================================

summary = pd.DataFrame(
    [
        {
            "metric": "total_cases",
            "value": n,
        },
        {
            "metric": "positive_cases",
            "value": positive_cases,
        },
        {
            "metric": "negative_cases",
            "value": negative_cases,
        },
        {
            "metric": "correct",
            "value": correct,
        },
        {
            "metric": "incorrect",
            "value": incorrect,
        },
        {
            "metric": "accuracy",
            "value": accuracy,
        },
        {
            "metric": "true_positive",
            "value": true_positive,
        },
        {
            "metric": "true_negative",
            "value": true_negative,
        },
        {
            "metric": "false_positive",
            "value": false_positive,
        },
        {
            "metric": "false_negative",
            "value": false_negative,
        },
        {
            "metric": "precision",
            "value": precision,
        },
        {
            "metric": "recall",
            "value": recall,
        },
        {
            "metric": "specificity",
            "value": specificity,
        },
        {
            "metric": "f1",
            "value": f1,
        },
        {
            "metric": "brier_score",
            "value": brier_score,
        },
        {
            "metric": "mean_absolute_probability_error",
            "value":
                mean_absolute_probability_error,
        },
        {
            "metric": "log_loss",
            "value": log_loss,
        },
        {
            "metric": "mean_prediction_confidence",
            "value": mean_confidence,
        },
        {
            "metric": "min_prediction_confidence",
            "value": min_confidence,
        },
        {
            "metric": "max_prediction_confidence",
            "value": max_confidence,
        },
        {
            "metric": "true_probability_mean",
            "value": true_probability_mean,
        },
        {
            "metric": "true_probability_median",
            "value": true_probability_median,
        },
        {
            "metric": "true_probability_min",
            "value": true_probability_min,
        },
        {
            "metric": "true_probability_max",
            "value": true_probability_max,
        },
        {
            "metric": "false_probability_mean",
            "value": false_probability_mean,
        },
        {
            "metric": "false_probability_median",
            "value": false_probability_median,
        },
        {
            "metric": "false_probability_min",
            "value": false_probability_min,
        },
        {
            "metric": "false_probability_max",
            "value": false_probability_max,
        },
        {
            "metric": "probability_separation",
            "value": probability_separation,
        },
        {
            "metric": "ece_5_bins",
            "value": ece_5,
        },
        {
            "metric": "ece_10_bins",
            "value": ece_10,
        },
        {
            "metric": "confidently_wrong_cases",
            "value": len(confident_wrong),
        },
        {
            "metric": "uncertain_cases_0.40_to_0.60",
            "value": len(uncertain_cases),
        },
    ]
)


summary.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ========================================================
# PRINT REPORT
# ========================================================

print()
print("=" * 60)
print("JEV PILOT ANALYSIS")
print("=" * 60)

print()
print("DATASET")
print("-" * 60)
print(f"Total cases:              {n}")
print(f"Positive cases:           {positive_cases}")
print(f"Negative cases:           {negative_cases}")

print()
print("CLASSIFICATION")
print("-" * 60)
print(f"Correct:                  {correct}")
print(f"Incorrect:                {incorrect}")
print(f"Accuracy:                 {accuracy:.4f}")
print(f"Precision:                {precision:.4f}")
print(f"Recall:                   {recall:.4f}")
print(f"Specificity:              {specificity:.4f}")
print(f"F1 score:                 {f1:.4f}")

print()
print("CONFUSION MATRIX")
print("-" * 60)
print(f"True Positive:            {true_positive}")
print(f"True Negative:            {true_negative}")
print(f"False Positive:           {false_positive}")
print(f"False Negative:           {false_negative}")

print()
print("PROBABILITY / RELIABILITY")
print("-" * 60)
print(f"Brier score:              {brier_score:.6f}")
print(
    "Mean absolute probability error: "
    f"{mean_absolute_probability_error:.6f}"
)
print(f"Log loss:                 {log_loss:.6f}")
print(f"ECE (5 bins):             {ece_5:.6f}")
print(f"ECE (10 bins):            {ece_10:.6f}")

print()
print("PREDICTION CONFIDENCE")
print("-" * 60)
print(f"Mean confidence:          {mean_confidence:.6f}")
print(f"Minimum confidence:       {min_confidence:.6f}")
print(f"Maximum confidence:       {max_confidence:.6f}")

print()
print("TRUE CASES")
print("-" * 60)
print(f"Mean probability:         {true_probability_mean:.6f}")
print(f"Median probability:       {true_probability_median:.6f}")
print(f"Minimum probability:      {true_probability_min:.6f}")
print(f"Maximum probability:      {true_probability_max:.6f}")

print()
print("FALSE CASES")
print("-" * 60)
print(f"Mean probability:         {false_probability_mean:.6f}")
print(f"Median probability:       {false_probability_median:.6f}")
print(f"Minimum probability:      {false_probability_min:.6f}")
print(f"Maximum probability:      {false_probability_max:.6f}")

print()
print("PROBABILITY SEPARATION")
print("-" * 60)
print(
    "Minimum true probability: "
    f"{true_probability_min:.6f}"
)
print(
    "Maximum false probability: "
    f"{false_probability_max:.6f}"
)
print(
    "Separation gap:           "
    f"{probability_separation:.6f}"
)

print()
print("RELIABILITY FLAGS")
print("-" * 60)
print(
    "Confidently wrong cases:  "
    f"{len(confident_wrong)}"
)
print(
    "Uncertain cases (0.40-0.60): "
    f"{len(uncertain_cases)}"
)

print()
print("DOMAIN ANALYSIS")
print("-" * 60)
print(domain_results.to_string(index=False))

print()
print("DIFFICULTY ANALYSIS")
print("-" * 60)
print(difficulty_results.to_string(index=False))

print()
print("OUTPUT")
print("-" * 60)
print(OUTPUT_FILE)

print()
print("=" * 60)