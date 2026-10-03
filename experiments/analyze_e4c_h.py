from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "experiments" / "data"
RESULTS = BASE / "experiments" / "results"

INPUT = DATA / "e4c_h_counterfactual_results.csv"

RESULTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

required = {
    "counterfactual_id",
    "original_id",
    "perturbation_type",
    "confidence_group",
    "original_probability",
    "label",
    "probability",
    "prediction",
    "correct",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

assert len(df) == 30
assert df["probability"].notna().all()
assert df["original_id"].nunique() == 10
assert df.groupby("original_id").size().eq(3).all()
assert df["perturbation_type"].value_counts().eq(10).all()

# Each counterfactual should inherit the original label.
assert (
    df.groupby("original_id")["label"]
    .nunique()
    .eq(1)
    .all()
)

# ============================================================
# CORE METRICS
# ============================================================

df["delta_p_signed"] = df["probability"] - df["original_probability"]
df["abs_delta_p"] = df["delta_p_signed"].abs()

df["decision_changed"] = (
    df["prediction"]
    != (df["original_probability"] >= 0.5).astype(int)
)

df["counterfactual_error"] = df["correct"] == 0


# ============================================================
# COUNTERFACTUAL-LEVEL SUMMARY
# ============================================================

overall = {
    "evaluations": len(df),
    "mean_abs_delta_p": df["abs_delta_p"].mean(),
    "median_abs_delta_p": df["abs_delta_p"].median(),
    "max_abs_delta_p": df["abs_delta_p"].max(),
    "counterfactual_errors": int(df["counterfactual_error"].sum()),
    "counterfactual_error_rate": df["counterfactual_error"].mean(),
    "decision_changes": int(df["decision_changed"].sum()),
}


print("=" * 70)
print("E4c-H COUNTERFACTUAL STABILITY ANALYSIS")
print("=" * 70)

print("\nCOUNTERFACTUAL-LEVEL RESULTS")
print(f"Evaluations:              {overall['evaluations']}")
print(f"Mean |delta p|:            {overall['mean_abs_delta_p']:.4f}")
print(f"Median |delta p|:          {overall['median_abs_delta_p']:.4f}")
print(f"Maximum |delta p|:         {overall['max_abs_delta_p']:.4f}")
print(f"Wrong decisions:           {overall['counterfactual_errors']}")
print(f"Error rate:                {overall['counterfactual_error_rate']:.4f}")
print(f"Decision changes:          {overall['decision_changes']}")


# ============================================================
# BY CONFIDENCE GROUP
# ============================================================

print("\n" + "=" * 70)
print("BY CONFIDENCE GROUP")
print("=" * 70)

group_summary = (
    df.groupby("confidence_group")
    .agg(
        evaluations=("counterfactual_id", "count"),
        originals=("original_id", "nunique"),
        mean_abs_delta_p=("abs_delta_p", "mean"),
        median_abs_delta_p=("abs_delta_p", "median"),
        max_abs_delta_p=("abs_delta_p", "max"),
        errors=("counterfactual_error", "sum"),
        error_rate=("counterfactual_error", "mean"),
    )
    .reset_index()
)

print(group_summary.to_string(index=False))


# ============================================================
# ORIGINAL-CASE LEVEL
# ============================================================

case_summary = (
    df.groupby(
        [
            "original_id",
            "confidence_group",
            "original_probability",
            "label",
        ],
        as_index=False,
    )
    .agg(
        mean_abs_delta_p=("abs_delta_p", "mean"),
        median_abs_delta_p=("abs_delta_p", "median"),
        max_abs_delta_p=("abs_delta_p", "max"),
        mean_signed_delta_p=("delta_p_signed", "mean"),
        min_probability=("probability", "min"),
        max_probability=("probability", "max"),
        errors=("counterfactual_error", "sum"),
        variants=("counterfactual_id", "count"),
    )
)

case_summary = case_summary.sort_values(
    "mean_abs_delta_p",
    ascending=False,
)

case_summary.to_csv(
    RESULTS / "e4c_h_case_summary.csv",
    index=False,
)

print("\n" + "=" * 70)
print("ORIGINAL-CASE LEVEL RESULTS")
print("=" * 70)

print(
    case_summary[
        [
            "original_id",
            "confidence_group",
            "original_probability",
            "mean_abs_delta_p",
            "max_abs_delta_p",
            "min_probability",
            "max_probability",
            "errors",
        ]
    ].to_string(index=False)
)


# ============================================================
# GROUP COMPARISON AT ORIGINAL-CASE LEVEL
# ============================================================

print("\n" + "=" * 70)
print("CASE-LEVEL GROUP COMPARISON")
print("=" * 70)

case_groups = (
    case_summary.groupby("confidence_group")
    .agg(
        originals=("original_id", "count"),
        mean_case_abs_delta=("mean_abs_delta_p", "mean"),
        median_case_abs_delta=("mean_abs_delta_p", "median"),
        max_case_abs_delta=("mean_abs_delta_p", "max"),
        total_errors=("errors", "sum"),
    )
    .reset_index()
)

print(case_groups.to_string(index=False))


# ============================================================
# PERTURBATION TYPE
# ============================================================

print("\n" + "=" * 70)
print("BY PERTURBATION TYPE")
print("=" * 70)

perturbation_summary = (
    df.groupby("perturbation_type")
    .agg(
        evaluations=("counterfactual_id", "count"),
        mean_abs_delta_p=("abs_delta_p", "mean"),
        median_abs_delta_p=("abs_delta_p", "median"),
        max_abs_delta_p=("abs_delta_p", "max"),
        errors=("counterfactual_error", "sum"),
        error_rate=("counterfactual_error", "mean"),
    )
    .reset_index()
)

print(perturbation_summary.to_string(index=False))


# ============================================================
# ERROR CASES
# ============================================================

errors = df[df["counterfactual_error"]].copy()

print("\n" + "=" * 70)
print("COUNTERFACTUAL ERRORS")
print("=" * 70)

if len(errors) == 0:
    print("No counterfactual errors.")
else:
    print(
        errors[
            [
                "counterfactual_id",
                "original_id",
                "perturbation_type",
                "confidence_group",
                "original_probability",
                "probability",
                "label",
                "prediction",
                "abs_delta_p",
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE ALL ANALYSIS DATA
# ============================================================

df.to_csv(
    RESULTS / "e4c_h_counterfactual_analysis.csv",
    index=False,
)

group_summary.to_csv(
    RESULTS / "e4c_h_group_summary.csv",
    index=False,
)

perturbation_summary.to_csv(
    RESULTS / "e4c_h_perturbation_summary.csv",
    index=False,
)

print("\n" + "=" * 70)
print("FILES WRITTEN")
print("=" * 70)

print("experiments/results/e4c_h_counterfactual_analysis.csv")
print("experiments/results/e4c_h_case_summary.csv")
print("experiments/results/e4c_h_group_summary.csv")
print("experiments/results/e4c_h_perturbation_summary.csv")

print("\nANALYSIS COMPLETE")
