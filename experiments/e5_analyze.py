import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("experiments/data")

original = pd.read_csv(BASE / "e5_original_results.csv")
adv = pd.read_csv(BASE / "e5_adversarial_results.csv")

# ------------------------------------------------------------
# 1. BASIC VALIDATION
# ------------------------------------------------------------

assert len(original) == 60
assert len(adv) == 180
assert original["original_id"].is_unique
assert set(adv["original_id"]) == set(original["original_id"])

# Every original must have exactly 3 variants
counts = adv.groupby("original_id").size()
assert (counts == 3).all()

# ------------------------------------------------------------
# 2. JOIN ORIGINAL -> ADVERSARIAL
# ------------------------------------------------------------

orig_cols = [
    "original_id",
    "probability",
    "prediction",
    "label",
    "correct",
]

orig = original[orig_cols].rename(
    columns={
        "probability": "original_probability",
        "prediction": "original_prediction",
        "label": "original_label",
        "correct": "original_correct",
    }
)

df = adv.merge(orig, on="original_id", how="left")

assert df["original_probability"].notna().all()

# ------------------------------------------------------------
# 3. PROBABILITY / CONFIDENCE METRICS
# ------------------------------------------------------------

df["delta_p"] = df["probability"] - df["original_probability"]
df["abs_delta_p"] = df["delta_p"].abs()

# Confidence = probability assigned to the actual label
df["original_confidence"] = np.where(
    df["original_label"] == 1,
    df["original_probability"],
    1 - df["original_probability"],
)

df["variant_confidence"] = np.where(
    df["label"] == 1,
    df["probability"],
    1 - df["probability"],
)

df["confidence_loss"] = (
    df["original_confidence"] - df["variant_confidence"]
)

df["decision_changed"] = (
    df["prediction"] != df["original_prediction"]
)

df["adversarial_error"] = (
    df["correct"] == 0
)

# ------------------------------------------------------------
# 4. OVERALL RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("E5 ADVERSARIAL ROBUSTNESS ANALYSIS")
print("=" * 70)

print("\nDATASET")
print(f"Original cases:       {len(original)}")
print(f"Counterfactuals:      {len(adv)}")
print(f"Domains:              {original['domain'].nunique()}")
print(f"Perturbation types:   {adv['perturbation_type'].nunique()}")

print("\nORIGINAL PERFORMANCE")
print(f"Accuracy:             {original['correct'].mean():.4f}")
print(
    f"Mean probability:     {original['probability'].mean():.4f}"
)
print(
    f"Median probability:   {original['probability'].median():.4f}"
)

print("\nCOUNTERFACTUAL PERFORMANCE")
print(f"Accuracy:             {adv['correct'].mean():.4f}")
print(
    f"Adversarial errors:   {(~adv['correct'].astype(bool)).sum()}/{len(adv)}"
)
print(
    f"Adversarial error rate: {(~adv['correct'].astype(bool)).mean():.4f}"
)
print(
    f"Decision changes:     {df['decision_changed'].sum()}/{len(df)}"
)
print(
    f"Decision-change rate: {df['decision_changed'].mean():.4f}"
)

print("\nPROBABILITY STABILITY")
print(f"Mean Δp:              {df['delta_p'].mean():.4f}")
print(f"Mean |Δp|:            {df['abs_delta_p'].mean():.4f}")
print(f"Median |Δp|:          {df['abs_delta_p'].median():.4f}")
print(f"Maximum |Δp|:         {df['abs_delta_p'].max():.4f}")

print("\nCONFIDENCE")
print(
    f"Mean original confidence: {df['original_confidence'].mean():.4f}"
)
print(
    f"Mean variant confidence:  {df['variant_confidence'].mean():.4f}"
)
print(
    f"Mean confidence loss:     {df['confidence_loss'].mean():.4f}"
)

# ------------------------------------------------------------
# 5. ERROR CASES
# ------------------------------------------------------------

errors = df[df["adversarial_error"]].copy()

print("\n" + "-" * 70)
print("ADVERSARIAL ERRORS")
print("-" * 70)

if len(errors) == 0:
    print("No adversarial errors observed.")
else:
    for _, r in errors.iterrows():
        print(
            f"{r['counterfactual_id']} | "
            f"original={r['original_id']} | "
            f"domain={r['domain']} | "
            f"type={r['perturbation_type']} | "
            f"original_p={r['original_probability']:.2f} | "
            f"variant_p={r['probability']:.2f} | "
            f"label={r['label']}"
        )

# ------------------------------------------------------------
# 6. PERTURBATION-TYPE EFFECTS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("PERTURBATION TYPE")
print("-" * 70)

ptype = (
    df.groupby("perturbation_type")
    .agg(
        n=("probability", "size"),
        accuracy=("correct", "mean"),
        mean_abs_delta_p=("abs_delta_p", "mean"),
        median_abs_delta_p=("abs_delta_p", "median"),
        max_abs_delta_p=("abs_delta_p", "max"),
        decision_change_rate=("decision_changed", "mean"),
        mean_confidence_loss=("confidence_loss", "mean"),
        error_rate=("adversarial_error", "mean"),
    )
    .sort_index()
)

print(ptype.to_string(float_format=lambda x: f"{x:.4f}"))

# ------------------------------------------------------------
# 7. DOMAIN EFFECTS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("DOMAIN EFFECTS")
print("-" * 70)

domain = (
    df.groupby("domain")
    .agg(
        n=("probability", "size"),
        accuracy=("correct", "mean"),
        mean_abs_delta_p=("abs_delta_p", "mean"),
        median_abs_delta_p=("abs_delta_p", "median"),
        max_abs_delta_p=("abs_delta_p", "max"),
        decision_change_rate=("decision_changed", "mean"),
        mean_confidence_loss=("confidence_loss", "mean"),
        error_rate=("adversarial_error", "mean"),
    )
    .sort_index()
)

print(domain.to_string(float_format=lambda x: f"{x:.4f}"))

# ------------------------------------------------------------
# 8. CASE-LEVEL SUMMARY
# ------------------------------------------------------------

case = (
    df.groupby("original_id")
    .agg(
        domain=("domain", "first"),
        original_probability=("original_probability", "first"),
        original_prediction=("original_prediction", "first"),
        label=("label", "first"),
        original_correct=("original_correct", "first"),
        mean_abs_delta_p=("abs_delta_p", "mean"),
        max_abs_delta_p=("abs_delta_p", "max"),
        mean_confidence_loss=("confidence_loss", "mean"),
        max_confidence_loss=("confidence_loss", "max"),
        decision_changes=("decision_changed", "sum"),
        adversarial_errors=("adversarial_error", "sum"),
    )
    .reset_index()
)

case["original_confidence"] = np.where(
    case["label"] == 1,
    case["original_probability"],
    1 - case["original_probability"],
)

# ------------------------------------------------------------
# 9. LOW/MODERATE/HIGH CONFIDENCE ANALYSIS
# ------------------------------------------------------------

# These bins are descriptive, not tuned to the observed results.
case["confidence_group"] = pd.cut(
    case["original_confidence"],
    bins=[-np.inf, 0.75, 0.90, np.inf],
    labels=["low/moderate", "high", "very_high"],
)

print("\n" + "-" * 70)
print("CASE-LEVEL CONFIDENCE GROUPS")
print("-" * 70)

confidence = (
    case.groupby("confidence_group", observed=False)
    .agg(
        cases=("original_id", "size"),
        mean_original_confidence=("original_confidence", "mean"),
        mean_abs_delta_p=("mean_abs_delta_p", "mean"),
        max_abs_delta_p=("max_abs_delta_p", "mean"),
        mean_confidence_loss=("mean_confidence_loss", "mean"),
        cases_with_decision_change=(
            "decision_changes",
            lambda x: (x > 0).sum(),
        ),
        cases_with_adversarial_error=(
            "adversarial_errors",
            lambda x: (x > 0).sum(),
        ),
    )
)

print(
    confidence.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# 10. CORRELATION: CONFIDENCE VS INSTABILITY
# ------------------------------------------------------------

pearson = case["original_confidence"].corr(
    case["mean_abs_delta_p"],
    method="pearson",
)

spearman = case["original_confidence"].corr(
    case["mean_abs_delta_p"],
    method="spearman",
)

print("\n" + "-" * 70)
print("CONFIDENCE vs PROBABILITY INSTABILITY")
print("-" * 70)
print(f"Pearson r:             {pearson:.4f}")
print(f"Spearman rho:          {spearman:.4f}")

# ------------------------------------------------------------
# 11. CASES WITH LARGEST INSTABILITY
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("TOP 10 CASES BY MEAN |Δp|")
print("-" * 70)

top = case.sort_values(
    "mean_abs_delta_p",
    ascending=False
).head(10)

print(
    top[
        [
            "original_id",
            "domain",
            "original_probability",
            "original_confidence",
            "mean_abs_delta_p",
            "max_abs_delta_p",
            "decision_changes",
            "adversarial_errors",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# 12. ERROR CONCENTRATION
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("ERROR CONCENTRATION")
print("-" * 70)

print(
    f"Variant-level errors: "
    f"{int(df['adversarial_error'].sum())}/{len(df)} "
    f"= {df['adversarial_error'].mean():.4f}"
)

affected_cases = (case["adversarial_errors"] > 0).sum()

print(
    f"Original cases with >=1 error: "
    f"{affected_cases}/{len(case)} "
    f"= {affected_cases / len(case):.4f}"
)

if affected_cases:
    error_cases = case[case["adversarial_errors"] > 0]
    print("\nAffected original cases:")
    print(
        error_cases[
            [
                "original_id",
                "domain",
                "original_probability",
                "original_confidence",
                "mean_abs_delta_p",
                "max_abs_delta_p",
                "adversarial_errors",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

# ------------------------------------------------------------
# 13. SAVE JOINED DATA
# ------------------------------------------------------------

df.to_csv(BASE / "e5_joined_results.csv", index=False)
case.to_csv(BASE / "e5_case_level_results.csv", index=False)
ptype.to_csv(BASE / "e5_perturbation_summary.csv")
domain.to_csv(BASE / "e5_domain_summary.csv")
confidence.to_csv(BASE / "e5_confidence_summary.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("Saved:")
print(BASE / "e5_joined_results.csv")
print(BASE / "e5_case_level_results.csv")
print(BASE / "e5_perturbation_summary.csv")
print(BASE / "e5_domain_summary.csv")
print(BASE / "e5_confidence_summary.csv")
