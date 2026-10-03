import os
import numpy as np
import pandas as pd


RESULTS_FILE = "experiments/data/e6_domain_shift_results.csv"
OUTPUT_DIR = "experiments/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BOOTSTRAP_ITERATIONS = 10_000
RANDOM_SEED = 42


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(RESULTS_FILE)

required_columns = {
    "case_id",
    "core_id",
    "shift_group",
    "source_domain",
    "target_domain",
    "domain",
    "label",
    "probability",
    "prediction",
    "correct",
}

missing = required_columns - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )


# Make sure numeric columns are numeric.
for column in [
    "label",
    "probability",
    "prediction",
    "correct",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

assert len(df) == 100
assert df["core_id"].nunique() == 50

assert (
    df.groupby("core_id")
    .size()
    .eq(2)
    .all()
)

assert set(df["shift_group"]) == {
    "source",
    "shifted",
}

assert df["probability"].notna().all()
assert df["prediction"].notna().all()
assert df["correct"].notna().all()

print("=" * 70)
print("E6 DOMAIN-SHIFT ANALYSIS")
print("=" * 70)

print()
print("✓ 100 results loaded")
print("✓ 50 matched pairs")
print("✓ No missing probabilities")
print("✓ No missing predictions")
print("✓ No missing correctness values")


# ============================================================
# RESHAPE INTO MATCHED PAIRS
# ============================================================

source = (
    df[df["shift_group"] == "source"]
    .copy()
    .set_index("core_id")
)

shifted = (
    df[df["shift_group"] == "shifted"]
    .copy()
    .set_index("core_id")
)


pairs = pd.DataFrame(index=source.index)

pairs["source_domain"] = source["source_domain"]
pairs["target_domain"] = source["target_domain"]

pairs["label"] = source["label"]

pairs["source_probability"] = source["probability"]
pairs["shifted_probability"] = shifted["probability"]

pairs["source_prediction"] = source["prediction"]
pairs["shifted_prediction"] = shifted["prediction"]

pairs["source_correct"] = source["correct"]
pairs["shifted_correct"] = shifted["correct"]


# Probability change:
pairs["delta_probability"] = (
    pairs["shifted_probability"]
    - pairs["source_probability"]
)

pairs["abs_delta_probability"] = (
    pairs["delta_probability"]
    .abs()
)

# Binary decision stability:
pairs["decision_changed"] = (
    pairs["source_prediction"]
    != pairs["shifted_prediction"]
).astype(int)

# Confidence:
pairs["source_confidence"] = np.maximum(
    pairs["source_probability"],
    1 - pairs["source_probability"],
)

pairs["shifted_confidence"] = np.maximum(
    pairs["shifted_probability"],
    1 - pairs["shifted_probability"],
)

pairs["confidence_change"] = (
    pairs["shifted_confidence"]
    - pairs["source_confidence"]
)

pairs["abs_confidence_change"] = (
    pairs["confidence_change"]
    .abs()
)


# ============================================================
# OVERALL METRICS
# ============================================================

source_accuracy = pairs["source_correct"].mean()
shifted_accuracy = pairs["shifted_correct"].mean()

mean_abs_delta = (
    pairs["abs_delta_probability"].mean()
)

median_abs_delta = (
    pairs["abs_delta_probability"].median()
)

max_abs_delta = (
    pairs["abs_delta_probability"].max()
)

decision_change_rate = (
    pairs["decision_changed"].mean()
)

mean_delta = (
    pairs["delta_probability"].mean()
)

mean_confidence_change = (
    pairs["confidence_change"].mean()
)

mean_abs_confidence_change = (
    pairs["abs_confidence_change"].mean()
)


print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Source accuracy:                 "
    f"{source_accuracy:.4f}"
)

print(
    f"Shifted accuracy:                "
    f"{shifted_accuracy:.4f}"
)

print(
    f"Mean probability change:         "
    f"{mean_delta:.4f}"
)

print(
    f"Mean absolute probability change:"
    f" {mean_abs_delta:.4f}"
)

print(
    f"Median absolute probability change:"
    f" {median_abs_delta:.4f}"
)

print(
    f"Maximum absolute probability change:"
    f" {max_abs_delta:.4f}"
)

print(
    f"Decision-change rate:             "
    f"{decision_change_rate:.4f}"
)

print(
    f"Mean confidence change:           "
    f"{mean_confidence_change:.4f}"
)

print(
    f"Mean absolute confidence change:  "
    f"{mean_abs_confidence_change:.4f}"
)


# ============================================================
# DOMAIN-PAIR ANALYSIS
# ============================================================

domain_summary = (
    pairs
    .groupby(
        ["source_domain", "target_domain"]
    )
    .agg(
        pairs=("label", "size"),
        source_accuracy=("source_correct", "mean"),
        shifted_accuracy=("shifted_correct", "mean"),
        mean_abs_probability_change=(
            "abs_delta_probability",
            "mean",
        ),
        median_abs_probability_change=(
            "abs_delta_probability",
            "median",
        ),
        max_abs_probability_change=(
            "abs_delta_probability",
            "max",
        ),
        decision_change_rate=(
            "decision_changed",
            "mean",
        ),
        mean_confidence_change=(
            "confidence_change",
            "mean",
        ),
    )
    .reset_index()
)


print()
print("=" * 70)
print("DOMAIN-SHIFT RESULTS")
print("=" * 70)

print(
    domain_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# DECISION CHANGES
# ============================================================

decision_changes = pairs[
    pairs["decision_changed"] == 1
].copy()


print()
print("=" * 70)
print("BINARY DECISION CHANGES")
print("=" * 70)

print(
    f"Decision changes: "
    f"{len(decision_changes)} / {len(pairs)}"
)

if len(decision_changes) > 0:

    print()

    print(
        decision_changes[
            [
                "source_domain",
                "target_domain",
                "label",
                "source_probability",
                "shifted_probability",
                "source_prediction",
                "shifted_prediction",
            ]
        ].to_string(
            index=True,
            float_format=lambda x: f"{x:.4f}",
        )
    )

else:

    print("No binary decisions changed.")


# ============================================================
# LARGEST PROBABILITY SHIFTS
# ============================================================

largest_shifts = (
    pairs
    .sort_values(
        "abs_delta_probability",
        ascending=False,
    )
    .head(10)
)


print()
print("=" * 70)
print("10 LARGEST PROBABILITY SHIFTS")
print("=" * 70)

print(
    largest_shifts[
        [
            "source_domain",
            "target_domain",
            "label",
            "source_probability",
            "shifted_probability",
            "delta_probability",
            "abs_delta_probability",
            "decision_changed",
        ]
    ].to_string(
        index=True,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# BOOTSTRAP
#
# Resample the 50 MATCHED PAIRS, not the 100 individual rows.
# This preserves the paired experimental structure.
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

n_pairs = len(pairs)

bootstrap_mean_abs_delta = np.empty(
    BOOTSTRAP_ITERATIONS
)

bootstrap_decision_change = np.empty(
    BOOTSTRAP_ITERATIONS
)

bootstrap_source_accuracy = np.empty(
    BOOTSTRAP_ITERATIONS
)

bootstrap_shifted_accuracy = np.empty(
    BOOTSTRAP_ITERATIONS
)

bootstrap_mean_confidence_change = np.empty(
    BOOTSTRAP_ITERATIONS
)


pair_indices = np.arange(n_pairs)

for i in range(BOOTSTRAP_ITERATIONS):

    sample_indices = rng.choice(
        pair_indices,
        size=n_pairs,
        replace=True,
    )

    sample = pairs.iloc[sample_indices]

    bootstrap_mean_abs_delta[i] = (
        sample["abs_delta_probability"].mean()
    )

    bootstrap_decision_change[i] = (
        sample["decision_changed"].mean()
    )

    bootstrap_source_accuracy[i] = (
        sample["source_correct"].mean()
    )

    bootstrap_shifted_accuracy[i] = (
        sample["shifted_correct"].mean()
    )

    bootstrap_mean_confidence_change[i] = (
        sample["confidence_change"].mean()
    )


def ci(values):

    return (
        np.percentile(values, 2.5),
        np.percentile(values, 97.5),
    )


mean_abs_delta_ci = ci(
    bootstrap_mean_abs_delta
)

decision_change_ci = ci(
    bootstrap_decision_change
)

source_accuracy_ci = ci(
    bootstrap_source_accuracy
)

shifted_accuracy_ci = ci(
    bootstrap_shifted_accuracy
)

confidence_change_ci = ci(
    bootstrap_mean_confidence_change
)


# ============================================================
# BOOTSTRAP RESULTS
# ============================================================

bootstrap_summary = pd.DataFrame(
    [
        {
            "metric": "source_accuracy",
            "estimate": source_accuracy,
            "ci_lower": source_accuracy_ci[0],
            "ci_upper": source_accuracy_ci[1],
        },
        {
            "metric": "shifted_accuracy",
            "estimate": shifted_accuracy,
            "ci_lower": shifted_accuracy_ci[0],
            "ci_upper": shifted_accuracy_ci[1],
        },
        {
            "metric": "mean_abs_probability_change",
            "estimate": mean_abs_delta,
            "ci_lower": mean_abs_delta_ci[0],
            "ci_upper": mean_abs_delta_ci[1],
        },
        {
            "metric": "decision_change_rate",
            "estimate": decision_change_rate,
            "ci_lower": decision_change_ci[0],
            "ci_upper": decision_change_ci[1],
        },
        {
            "metric": "mean_confidence_change",
            "estimate": mean_confidence_change,
            "ci_lower": confidence_change_ci[0],
            "ci_upper": confidence_change_ci[1],
        },
    ]
)


print()
print("=" * 70)
print("10,000-PAIR BOOTSTRAP — 95% CI")
print("=" * 70)

print(
    bootstrap_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

pairs_file = (
    f"{OUTPUT_DIR}/e6_matched_pairs.csv"
)

domain_file = (
    f"{OUTPUT_DIR}/e6_domain_shift_summary.csv"
)

bootstrap_file = (
    f"{OUTPUT_DIR}/e6_bootstrap_summary.csv"
)

largest_file = (
    f"{OUTPUT_DIR}/e6_largest_probability_shifts.csv"
)

pairs.reset_index().to_csv(
    pairs_file,
    index=False,
)

domain_summary.to_csv(
    domain_file,
    index=False,
)

bootstrap_summary.to_csv(
    bootstrap_file,
    index=False,
)

largest_shifts.reset_index().to_csv(
    largest_file,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("E6 ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Files written:")

print(pairs_file)
print(domain_file)
print(bootstrap_file)
print(largest_file)

print()
print("No additional API calls were made.")
print("Bootstrap unit: 50 matched pairs.")
