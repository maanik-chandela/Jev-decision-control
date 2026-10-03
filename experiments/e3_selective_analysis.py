import numpy as np
import pandas as pd


INPUT_FILE = "experiments/data/e3_selective_prediction_results.csv"

RELIABILITY_FILE = "experiments/data/e3_selective_reliability.csv"
THRESHOLD_FILE = "experiments/data/e3_threshold_summary.csv"
COVERAGE_FILE = "experiments/data/e3_risk_coverage.csv"
BOOTSTRAP_FILE = "experiments/data/e3_bootstrap_summary.csv"


# Pre-specified thresholds.
# These are fixed before looking at the evaluation results.
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def confidence_from_probability(p):
    return max(p, 1.0 - p)


def compute_threshold_metrics(df, threshold):
    """
    Auto-answer cases whose confidence >= threshold.
    All remaining cases are abstained from.
    """

    selected = df["confidence"] >= threshold

    n_total = len(df)
    n_selected = int(selected.sum())
    n_abstained = n_total - n_selected

    coverage = n_selected / n_total
    abstention_rate = n_abstained / n_total

    if n_selected > 0:
        selective_risk = 1.0 - df.loc[selected, "correct"].mean()
        selective_accuracy = df.loc[selected, "correct"].mean()
    else:
        selective_risk = np.nan
        selective_accuracy = np.nan

    errors = int(((selected) & (df["correct"] == 0)).sum())

    return {
        "threshold": threshold,
        "n_total": n_total,
        "n_auto_answered": n_selected,
        "n_abstained": n_abstained,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "selective_accuracy": selective_accuracy,
        "selective_risk": selective_risk,
        "errors": errors,
    }


def compute_risk_coverage(df):
    """
    Risk-coverage curve.

    Cases are ordered from highest confidence to lowest confidence.
    At each coverage level, calculate error rate among accepted cases.
    """

    ordered = df.sort_values(
        "confidence",
        ascending=False
    ).reset_index(drop=True)

    rows = []

    cumulative_errors = 0

    for i, row in ordered.iterrows():

        cumulative_errors += int(row["correct"] == 0)

        n = i + 1
        coverage = n / len(ordered)
        risk = cumulative_errors / n

        rows.append({
            "n_auto_answered": n,
            "coverage": coverage,
            "selective_risk": risk,
            "selective_accuracy": 1.0 - risk,
            "confidence_cutoff": row["confidence"],
        })

    return pd.DataFrame(rows)


def compute_aurc(risk_coverage):
    """
    Approximate area under the risk-coverage curve using
    the trapezoidal rule.
    """

    x = risk_coverage["coverage"].to_numpy()
    y = risk_coverage["selective_risk"].to_numpy()

    return float(np.trapezoid(y, x))


def bootstrap_threshold_metrics(
    df,
    thresholds,
    n_bootstrap=10000,
    seed=42,
):
    """
    Bootstrap over independent cases.
    """

    rng = np.random.default_rng(seed)

    n = len(df)

    estimates = {
        threshold: []
        for threshold in thresholds
    }

    aurcs = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample = df.iloc[indices].reset_index(drop=True)

        for threshold in thresholds:
            metrics = compute_threshold_metrics(
                sample,
                threshold,
            )

            estimates[threshold].append(
                metrics["selective_risk"]
            )

        rc = compute_risk_coverage(sample)
        aurcs.append(compute_aurc(rc))

    rows = []

    for threshold in thresholds:

        values = np.array(
            [
                x
                for x in estimates[threshold]
                if not np.isnan(x)
            ]
        )

        rows.append({
            "metric": "selective_risk",
            "threshold": threshold,
            "estimate": compute_threshold_metrics(
                df,
                threshold,
            )["selective_risk"],
            "ci_lower": np.percentile(values, 2.5),
            "ci_upper": np.percentile(values, 97.5),
        })

    rows.append({
        "metric": "AURC",
        "threshold": np.nan,
        "estimate": compute_aurc(
            compute_risk_coverage(df)
        ),
        "ci_lower": np.percentile(
            aurcs,
            2.5,
        ),
        "ci_upper": np.percentile(
            aurcs,
            97.5,
        ),
    })

    return pd.DataFrame(rows)


def main():

    df = pd.read_csv(INPUT_FILE)

    print("=" * 60)
    print("E3 SELECTIVE PREDICTION ANALYSIS")
    print("=" * 60)

    df["confidence"] = df["probability"].apply(
        confidence_from_probability
    )

    df["error"] = 1 - df["correct"]

    print("\nBASIC PERFORMANCE")
    print("-" * 60)

    print(f"Cases:    {len(df)}")
    print(f"Accuracy: {df['correct'].mean():.4f}")
    print(
        f"Mean confidence: "
        f"{df['confidence'].mean():.4f}"
    )

    print(
        f"Mean confidence (correct): "
        f"{df.loc[df['correct'] == 1, 'confidence'].mean():.4f}"
    )

    print(
        f"Mean confidence (incorrect): "
        f"{df.loc[df['correct'] == 0, 'confidence'].mean():.4f}"
    )

    # ---------------------------------------------------------
    # Fixed thresholds
    # ---------------------------------------------------------

    threshold_rows = []

    for threshold in THRESHOLDS:

        metrics = compute_threshold_metrics(
            df,
            threshold,
        )

        threshold_rows.append(metrics)

    threshold_df = pd.DataFrame(threshold_rows)

    threshold_df.to_csv(
        THRESHOLD_FILE,
        index=False,
    )

    print("\nFIXED THRESHOLDS")
    print("-" * 60)

    print(
        threshold_df[
            [
                "threshold",
                "n_auto_answered",
                "n_abstained",
                "coverage",
                "abstention_rate",
                "selective_accuracy",
                "selective_risk",
                "errors",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Risk-coverage curve
    # ---------------------------------------------------------

    rc = compute_risk_coverage(df)

    rc.to_csv(
        COVERAGE_FILE,
        index=False,
    )

    aurc = compute_aurc(rc)

    print("\nRISK-COVERAGE")
    print("-" * 60)

    print(f"AURC: {aurc:.6f}")

    # ---------------------------------------------------------
    # Reliability-style confidence bins
    # ---------------------------------------------------------

    bins = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        1.00,
    ]

    labels = [
        "0.50-0.59",
        "0.60-0.69",
        "0.70-0.79",
        "0.80-0.89",
        "0.90-0.94",
        "0.95-0.99",
    ]

    df["confidence_bin"] = pd.cut(
        df["confidence"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    reliability = (
        df.groupby(
            "confidence_bin",
            observed=False,
        )
        .agg(
            n=("correct", "size"),
            mean_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
            error_rate=("error", "mean"),
        )
        .reset_index()
    )

    reliability.to_csv(
        RELIABILITY_FILE,
        index=False,
    )

    print("\nCONFIDENCE BINS")
    print("-" * 60)

    print(
        reliability.to_string(index=False)
    )

    # ---------------------------------------------------------
    # Bootstrap
    # ---------------------------------------------------------

    print("\nBOOTSTRAP")
    print("-" * 60)
    print("Running 10,000 case-level bootstrap resamples...")

    bootstrap_df = bootstrap_threshold_metrics(
        df,
        THRESHOLDS,
        n_bootstrap=10000,
        seed=42,
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_FILE,
        index=False,
    )

    print(
        bootstrap_df.to_string(index=False)
    )

    # ---------------------------------------------------------
    # Error cases
    # ---------------------------------------------------------

    print("\nINCORRECT CASES")
    print("-" * 60)

    print(
        df.loc[
            df["correct"] == 0,
            [
                "original_id",
                "domain",
                "difficulty",
                "label",
                "probability",
                "confidence",
                "prediction",
            ],
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {THRESHOLD_FILE}")
    print(f"  {COVERAGE_FILE}")
    print(f"  {RELIABILITY_FILE}")
    print(f"  {BOOTSTRAP_FILE}")


if __name__ == "__main__":
    main()
