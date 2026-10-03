import numpy as np
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

INPUT = (
    BASE
    / "experiments"
    / "data"
    / "e4b_counterfactual_results.csv"
)

RESULTS = BASE / "experiments" / "results"

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(INPUT)


# ------------------------------------------------------------
# Basic derived variables
# ------------------------------------------------------------

df["baseline_probability"] = df[
    "baseline_probability"
].astype(float)

df["jev_probability"] = df[
    "jev_probability"
].astype(float)

df["absolute_delta_p"] = (
    df["jev_probability"]
    - df["baseline_probability"]
).abs()

df["baseline_uncertainty"] = (
    1
    - 2
    * (
        df["baseline_probability"]
        - 0.5
    ).abs()
)


# ------------------------------------------------------------
# Define confidence groups
# ------------------------------------------------------------

df["confidence_group"] = np.where(
    df["candidate_id"].isin(
        [
            "H002",
            "H003",
            "H007",
            "H009",
            "H013",
            "H021",
        ]
    ),
    "high_confidence",
    "moderate_low_confidence",
)


# ------------------------------------------------------------
# Bootstrap function
# ------------------------------------------------------------

def bootstrap_mean_difference(
    group_a,
    group_b,
    iterations=10000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    a = np.asarray(group_a)
    b = np.asarray(group_b)

    differences = []

    for _ in range(iterations):

        a_sample = rng.choice(
            a,
            size=len(a),
            replace=True,
        )

        b_sample = rng.choice(
            b,
            size=len(b),
            replace=True,
        )

        differences.append(
            np.mean(b_sample)
            - np.mean(a_sample)
        )

    differences = np.asarray(
        differences
    )

    return (
        np.mean(differences),
        np.percentile(
            differences,
            2.5
        ),
        np.percentile(
            differences,
            97.5
        ),
    )


# ------------------------------------------------------------
# Overall summary
# ------------------------------------------------------------

overall = {
    "n": len(df),
    "mean_absolute_delta_p": df[
        "absolute_delta_p"
    ].mean(),
    "median_absolute_delta_p": df[
        "absolute_delta_p"
    ].median(),
    "max_absolute_delta_p": df[
        "absolute_delta_p"
    ].max(),
    "decision_flip_count": int(
        df["decision_flip"].sum()
    ),
    "decision_flip_rate": df[
        "decision_flip"
    ].mean(),
    "correct_count": int(
        df["correct"].sum()
    ),
    "accuracy": df[
        "correct"
    ].mean(),
}


# ------------------------------------------------------------
# Confidence-group analysis
# ------------------------------------------------------------

group_summary = (
    df
    .groupby("confidence_group")
    .agg(
        n=(
            "absolute_delta_p",
            "count"
        ),
        mean_absolute_delta_p=(
            "absolute_delta_p",
            "mean"
        ),
        median_absolute_delta_p=(
            "absolute_delta_p",
            "median"
        ),
        max_absolute_delta_p=(
            "absolute_delta_p",
            "max"
        ),
        decision_flip_rate=(
            "decision_flip",
            "mean"
        ),
        accuracy=(
            "correct",
            "mean"
        ),
        mean_baseline_probability=(
            "baseline_probability",
            "mean"
        ),
    )
    .reset_index()
)


high = df.loc[
    df["confidence_group"]
    == "high_confidence",
    "absolute_delta_p"
]

moderate = df.loc[
    df["confidence_group"]
    == "moderate_low_confidence",
    "absolute_delta_p"
]


bootstrap_mean, bootstrap_low, bootstrap_high = (
    bootstrap_mean_difference(
        high,
        moderate
    )
)


high_mean = high.mean()
moderate_mean = moderate.mean()

ratio = (
    moderate_mean / high_mean
    if high_mean > 0
    else np.inf
)


# ------------------------------------------------------------
# Perturbation analysis
# ------------------------------------------------------------

perturbation_summary = (
    df
    .groupby(
        [
            "confidence_group",
            "perturbation_type",
        ]
    )
    .agg(
        n=(
            "absolute_delta_p",
            "count"
        ),
        mean_absolute_delta_p=(
            "absolute_delta_p",
            "mean"
        ),
        median_absolute_delta_p=(
            "absolute_delta_p",
            "median"
        ),
        max_absolute_delta_p=(
            "absolute_delta_p",
            "max"
        ),
        decision_flip_rate=(
            "decision_flip",
            "mean"
        ),
    )
    .reset_index()
)


# ------------------------------------------------------------
# Per-candidate analysis
# ------------------------------------------------------------

candidate_summary = (
    df
    .groupby(
        [
            "candidate_id",
            "confidence_group",
        ]
    )
    .agg(
        baseline_probability=(
            "baseline_probability",
            "first"
        ),
        n_variants=(
            "variant_id",
            "count"
        ),
        mean_absolute_delta_p=(
            "absolute_delta_p",
            "mean"
        ),
        median_absolute_delta_p=(
            "absolute_delta_p",
            "median"
        ),
        max_absolute_delta_p=(
            "absolute_delta_p",
            "max"
        ),
        decision_flip_rate=(
            "decision_flip",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_absolute_delta_p",
        ascending=False,
    )
)


# ------------------------------------------------------------
# Correlation with baseline uncertainty
# ------------------------------------------------------------

pearson = df[
    [
        "baseline_uncertainty",
        "absolute_delta_p",
    ]
].corr(
    method="pearson"
).iloc[0, 1]

spearman = df[
    [
        "baseline_uncertainty",
        "absolute_delta_p",
    ]
].corr(
    method="spearman"
).iloc[0, 1]


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

group_summary.to_csv(
    RESULTS / "e4b_confidence_groups.csv",
    index=False,
)

perturbation_summary.to_csv(
    RESULTS / "e4b_perturbation_summary.csv",
    index=False,
)

candidate_summary.to_csv(
    RESULTS / "e4b_candidate_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# Print report
# ------------------------------------------------------------

print("=" * 70)
print("E4b CONFIDENCE-STRATIFIED COUNTERFACTUAL ANALYSIS")
print("=" * 70)

print()

print("OVERALL")
print("-" * 70)

print(
    f"Evaluations: {overall['n']}"
)

print(
    f"Mean |delta p|: "
    f"{overall['mean_absolute_delta_p']:.4f}"
)

print(
    f"Median |delta p|: "
    f"{overall['median_absolute_delta_p']:.4f}"
)

print(
    f"Maximum |delta p|: "
    f"{overall['max_absolute_delta_p']:.4f}"
)

print(
    f"Decision flips: "
    f"{overall['decision_flip_count']}"
)

print(
    f"Decision flip rate: "
    f"{overall['decision_flip_rate']:.4f}"
)

print(
    f"Accuracy: "
    f"{overall['accuracy']:.4f}"
)

print()

print("CONFIDENCE GROUPS")
print("-" * 70)

print(
    group_summary.to_string(
        index=False
    )
)

print()

print("MODERATE/LOW vs HIGH")
print("-" * 70)

print(
    f"High-confidence mean |delta p|: "
    f"{high_mean:.4f}"
)

print(
    f"Moderate/low-confidence mean |delta p|: "
    f"{moderate_mean:.4f}"
)

print(
    f"Mean ratio: "
    f"{ratio:.2f}x"
)

print(
    f"Bootstrap mean difference "
    f"(moderate - high): "
    f"{bootstrap_mean:.4f}"
)

print(
    f"Bootstrap 95% CI: "
    f"[{bootstrap_low:.4f}, "
    f"{bootstrap_high:.4f}]"
)

print()

print("CORRELATION")
print("-" * 70)

print(
    f"Pearson correlation "
    f"(baseline uncertainty, |delta p|): "
    f"{pearson:.4f}"
)

print(
    f"Spearman correlation "
    f"(baseline uncertainty, |delta p|): "
    f"{spearman:.4f}"
)

print()

print("BY PERTURBATION TYPE")
print("-" * 70)

print(
    perturbation_summary.to_string(
        index=False
    )
)

print()

print("BY CANDIDATE")
print("-" * 70)

print(
    candidate_summary.to_string(
        index=False
    )
)

print()

print("=" * 70)
print("ANALYSIS FILES SAVED")
print("=" * 70)

print(
    RESULTS / "e4b_confidence_groups.csv"
)

print(
    RESULTS / "e4b_perturbation_summary.csv"
)

print(
    RESULTS / "e4b_candidate_summary.csv"
)
