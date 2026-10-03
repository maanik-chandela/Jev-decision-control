import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# E8 — Decision Representation Sensitivity Analysis
# ============================================================

INPUT = Path("experiments/data/e8_decision_representation_results.csv")
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAPS = 10_000
RANDOM_SEED = 42

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

print("=" * 70)
print("E8 — DECISION REPRESENTATION SENSITIVITY ANALYSIS")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Cases: {df['case_id'].nunique()}")

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

assert len(df) == 180
assert df["case_id"].nunique() == 60
assert df["row_id"].nunique() == 180

assert df["probability"].notna().all()
assert df["prediction"].notna().all()
assert df["correct"].notna().all()

assert df.groupby("case_id").size().eq(3).all()
assert df.groupby("case_id")["label"].nunique().eq(1).all()

expected_representations = [
    "classification",
    "direct",
    "predicate",
]

assert sorted(df["representation"].unique()) == expected_representations

print("\nValidation: PASSED")

# ------------------------------------------------------------
# Confidence
# ------------------------------------------------------------

df["confidence"] = np.where(
    df["label"] == 1,
    df["probability"],
    1.0 - df["probability"],
)

# ------------------------------------------------------------
# Representation summary
# ------------------------------------------------------------

summary_rows = []

for rep in expected_representations:

    sub = df[df["representation"] == rep]

    summary_rows.append({
        "representation": rep,
        "n": len(sub),
        "accuracy": sub["correct"].mean(),
        "mean_probability": sub["probability"].mean(),
        "median_probability": sub["probability"].median(),
        "mean_confidence": sub["confidence"].mean(),
        "median_confidence": sub["confidence"].median(),
    })

representation_summary = pd.DataFrame(summary_rows)

print("\n" + "-" * 70)
print("REPRESENTATION SUMMARY")
print("-" * 70)

print(
    representation_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

representation_summary.to_csv(
    RESULTS_DIR / "e8_representation_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# Pivot to one row per underlying case
# ------------------------------------------------------------

prob_pivot = df.pivot(
    index="case_id",
    columns="representation",
    values="probability",
)

prediction_pivot = df.pivot(
    index="case_id",
    columns="representation",
    values="prediction",
)

label_pivot = df.pivot(
    index="case_id",
    columns="representation",
    values="label",
)

confidence_pivot = df.pivot(
    index="case_id",
    columns="representation",
    values="confidence",
)

# ------------------------------------------------------------
# Pairwise probability shifts
# ------------------------------------------------------------

pairs = [
    ("direct", "predicate"),
    ("direct", "classification"),
    ("predicate", "classification"),
]

pair_rows = []

for a, b in pairs:

    signed = prob_pivot[b] - prob_pivot[a]
    absolute = signed.abs()

    decision_change = (
        prediction_pivot[a] != prediction_pivot[b]
    )

    pair_rows.append({
        "pair": f"{a}_vs_{b}",
        "mean_signed_change": signed.mean(),
        "mean_absolute_change": absolute.mean(),
        "median_absolute_change": absolute.median(),
        "max_absolute_change": absolute.max(),
        "decision_change_rate": decision_change.mean(),
        "decision_changes": int(decision_change.sum()),
    })

pairwise_summary = pd.DataFrame(pair_rows)

print("\n" + "-" * 70)
print("PAIRWISE PROBABILITY SHIFTS")
print("-" * 70)

print(
    pairwise_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

pairwise_summary.to_csv(
    RESULTS_DIR / "e8_pairwise_probability_shifts.csv",
    index=False,
)

# ------------------------------------------------------------
# Case-level sensitivity
# ------------------------------------------------------------

case_level = pd.DataFrame(index=prob_pivot.index)

case_level["probability_min"] = prob_pivot.min(axis=1)
case_level["probability_max"] = prob_pivot.max(axis=1)

case_level["probability_spread"] = (
    case_level["probability_max"]
    - case_level["probability_min"]
)

case_level["probability_std"] = prob_pivot.std(axis=1)

case_level["any_decision_change"] = (
    prediction_pivot.nunique(axis=1) > 1
).astype(int)

case_level["baseline_label"] = label_pivot["direct"]

case_level["domain"] = (
    df.groupby("case_id")["domain"]
    .first()
)

case_level["case_mean_confidence"] = (
    confidence_pivot.mean(axis=1)
)

case_level = case_level.reset_index()

print("\n" + "-" * 70)
print("CASE-LEVEL REPRESENTATION SENSITIVITY")
print("-" * 70)

print(
    f"Mean probability spread: "
    f"{case_level['probability_spread'].mean():.4f}"
)

print(
    f"Median probability spread: "
    f"{case_level['probability_spread'].median():.4f}"
)

print(
    f"Maximum probability spread: "
    f"{case_level['probability_spread'].max():.4f}"
)

print(
    f"Cases with any decision change: "
    f"{case_level['any_decision_change'].sum()}/60 "
    f"({case_level['any_decision_change'].mean():.4f})"
)

case_level.to_csv(
    RESULTS_DIR / "e8_case_level_sensitivity.csv",
    index=False,
)

# ------------------------------------------------------------
# Largest probability spreads
# ------------------------------------------------------------

largest = case_level.sort_values(
    "probability_spread",
    ascending=False,
).head(15)

largest.to_csv(
    RESULTS_DIR / "e8_largest_probability_spreads.csv",
    index=False,
)

print("\n" + "-" * 70)
print("LARGEST PROBABILITY SPREADS")
print("-" * 70)

print(
    largest[
        [
            "case_id",
            "domain",
            "baseline_label",
            "probability_min",
            "probability_max",
            "probability_spread",
            "any_decision_change",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# Domain-level analysis
# ------------------------------------------------------------

domain_rows = []

for domain, group in df.groupby("domain"):

    pivot = group.pivot(
        index="case_id",
        columns="representation",
        values="probability",
    )

    predictions = group.pivot(
        index="case_id",
        columns="representation",
        values="prediction",
    )

    spreads = pivot.max(axis=1) - pivot.min(axis=1)

    domain_rows.append({
        "domain": domain,
        "cases": pivot.shape[0],
        "mean_probability_spread": spreads.mean(),
        "median_probability_spread": spreads.median(),
        "max_probability_spread": spreads.max(),
        "any_decision_change_rate": (
            predictions.nunique(axis=1) > 1
        ).mean(),
    })

domain_summary = pd.DataFrame(domain_rows)

print("\n" + "-" * 70)
print("DOMAIN-LEVEL SENSITIVITY")
print("-" * 70)

print(
    domain_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

domain_summary.to_csv(
    RESULTS_DIR / "e8_domain_summary.csv",
    index=False,
)

# ============================================================
# BOOTSTRAP
# ============================================================
#
# IMPORTANT:
# Resample the 60 underlying cases.
# Each sampled case automatically carries its 3 representations.
#
# This implementation is vectorized and avoids slow pandas
# row-by-row lookups.
# ============================================================

print("\n" + "-" * 70)
print("BOOTSTRAP — 10,000 CASE-LEVEL RESAMPLES")
print("-" * 70)

rng = np.random.default_rng(RANDOM_SEED)

# Fixed ordering of the 60 cases
case_order = prob_pivot.index.to_numpy()

# Convert matrices to numpy arrays
P = prob_pivot.loc[case_order].to_numpy()

Y = label_pivot.loc[case_order, "direct"].to_numpy()

PRED = prediction_pivot.loc[
    case_order,
    ["direct", "predicate", "classification"]
].to_numpy()

# Pairwise absolute probability differences
direct_predicate = np.abs(P[:, 1] - P[:, 0])
direct_classification = np.abs(P[:, 2] - P[:, 0])
predicate_classification = np.abs(P[:, 2] - P[:, 1])

case_spread = P.max(axis=1) - P.min(axis=1)

case_any_decision_change = (
    np.max(PRED, axis=1) != np.min(PRED, axis=1)
).astype(float)

# Accuracy per representation
case_accuracy = (
    PRED == Y[:, None]
).astype(float)

# ------------------------------------------------------------
# Generate bootstrap samples
# Shape = 10,000 × 60
# ------------------------------------------------------------

sample_indices = rng.integers(
    low=0,
    high=len(case_order),
    size=(N_BOOTSTRAPS, len(case_order)),
)

# ------------------------------------------------------------
# Vectorized bootstrap metrics
# ------------------------------------------------------------

bootstrap_summary = pd.DataFrame({
    "mean_spread": case_spread[sample_indices].mean(axis=1),

    "median_spread": np.median(
        case_spread[sample_indices],
        axis=1
    ),

    "direct_predicate_abs_delta":
        direct_predicate[sample_indices].mean(axis=1),

    "direct_classification_abs_delta":
        direct_classification[sample_indices].mean(axis=1),

    "predicate_classification_abs_delta":
        predicate_classification[sample_indices].mean(axis=1),

    "any_decision_change_rate":
        case_any_decision_change[sample_indices].mean(axis=1),

    "direct_accuracy":
        case_accuracy[sample_indices, 0].mean(axis=1),

    "predicate_accuracy":
        case_accuracy[sample_indices, 1].mean(axis=1),

    "classification_accuracy":
        case_accuracy[sample_indices, 2].mean(axis=1),
})

bootstrap_summary.index.name = "iteration"

bootstrap_summary.to_csv(
    RESULTS_DIR / "e8_bootstrap_samples.csv"
)

# ------------------------------------------------------------
# Bootstrap confidence intervals
# ------------------------------------------------------------

bootstrap_metrics = []

for metric in bootstrap_summary.columns:

    values = bootstrap_summary[metric]

    bootstrap_metrics.append({
        "metric": metric,
        "estimate": values.mean(),
        "ci_lower": values.quantile(0.025),
        "ci_upper": values.quantile(0.975),
    })

bootstrap_ci_summary = pd.DataFrame(
    bootstrap_metrics
)

print(
    bootstrap_ci_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

bootstrap_ci_summary.to_csv(
    RESULTS_DIR / "e8_bootstrap_summary.csv",
    index=False,
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("E8 FINAL SUMMARY")
print("=" * 70)

print(
    f"Overall accuracy: "
    f"{df['correct'].mean():.4f}"
)

print(
    f"Mean probability spread: "
    f"{case_level['probability_spread'].mean():.4f}"
)

print(
    f"Median probability spread: "
    f"{case_level['probability_spread'].median():.4f}"
)

print(
    f"Maximum probability spread: "
    f"{case_level['probability_spread'].max():.4f}"
)

print(
    f"Any representation decision change: "
    f"{case_level['any_decision_change'].mean():.4f}"
)

print("\nFiles written:")

for filename in [
    "e8_representation_summary.csv",
    "e8_pairwise_probability_shifts.csv",
    "e8_case_level_sensitivity.csv",
    "e8_largest_probability_spreads.csv",
    "e8_domain_summary.csv",
    "e8_bootstrap_samples.csv",
    "e8_bootstrap_summary.csv",
]:
    print(f"  experiments/results/{filename}")

print("\nE8 analysis complete.")
