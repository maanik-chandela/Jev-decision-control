import numpy as np
import pandas as pd


INPUT_FILE = "experiments/data/e3_selective_prediction_results.csv"

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

N_BOOTSTRAP = 10000
SEED = 42


def confidence(p):
    return max(p, 1.0 - p)


def threshold_metrics(df, threshold):
    selected = df["confidence"] >= threshold

    n = int(selected.sum())
    errors = int(((selected) & (df["correct"] == 0)).sum())

    return {
        "threshold": threshold,
        "coverage": n / len(df),
        "abstention_rate": 1 - n / len(df),
        "n_auto_answered": n,
        "errors": errors,
        "selective_risk": errors / n if n > 0 else np.nan,
        "selective_accuracy": (
            1 - errors / n if n > 0 else np.nan
        ),
    }


def random_abstention_risk(
    df,
    coverage,
    rng,
):
    """
    Randomly select the same number of cases as
    the confidence-based controller and calculate risk.
    """

    n = len(df)
    k = int(round(coverage * n))

    risks = []

    for _ in range(N_BOOTSTRAP):
        selected_indices = rng.choice(
            n,
            size=k,
            replace=False,
        )

        selected = df.iloc[selected_indices]

        risk = 1 - selected["correct"].mean()
        risks.append(risk)

    return (
        float(np.mean(risks)),
        float(np.percentile(risks, 2.5)),
        float(np.percentile(risks, 97.5)),
    )


def risk_coverage(df):
    ordered = df.sort_values(
        "confidence",
        ascending=False,
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
        })

    return pd.DataFrame(rows)


def main():

    rng = np.random.default_rng(SEED)

    df = pd.read_csv(INPUT_FILE)

    df["confidence"] = df["probability"].apply(confidence)

    print("=" * 65)
    print("E3 FINAL SELECTIVE PREDICTION ANALYSIS")
    print("=" * 65)

    # ---------------------------------------------------------
    # Fixed thresholds + random baseline
    # ---------------------------------------------------------

    threshold_rows = []

    for threshold in THRESHOLDS:

        metrics = threshold_metrics(
            df,
            threshold,
        )

        (
            random_mean,
            random_lower,
            random_upper,
        ) = random_abstention_risk(
            df,
            metrics["coverage"],
            rng,
        )

        metrics["random_risk_mean"] = random_mean
        metrics["random_risk_ci_lower"] = random_lower
        metrics["random_risk_ci_upper"] = random_upper

        threshold_rows.append(metrics)

    threshold_df = pd.DataFrame(threshold_rows)

    print("\nTHRESHOLD VS RANDOM-ABSTENTION BASELINE")
    print("-" * 65)

    print(
        threshold_df[
            [
                "threshold",
                "coverage",
                "abstention_rate",
                "n_auto_answered",
                "errors",
                "selective_risk",
                "random_risk_mean",
                "random_risk_ci_lower",
                "random_risk_ci_upper",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Domain analysis
    # ---------------------------------------------------------

    domain_rows = []

    for domain, group in df.groupby("domain"):

        confidence_mean = group["confidence"].mean()
        accuracy = group["correct"].mean()

        row = {
            "domain": domain,
            "n": len(group),
            "accuracy": accuracy,
            "mean_confidence": confidence_mean,
            "errors": int((group["correct"] == 0).sum()),
        }

        for threshold in THRESHOLDS:

            metrics = threshold_metrics(
                group,
                threshold,
            )

            row[
                f"coverage_{threshold:.2f}"
            ] = metrics["coverage"]

            row[
                f"risk_{threshold:.2f}"
            ] = metrics["selective_risk"]

        domain_rows.append(row)

    domain_df = pd.DataFrame(domain_rows)

    domain_df.to_csv(
        "experiments/data/e3_domain_summary.csv",
        index=False,
    )

    print("\nDOMAIN ANALYSIS")
    print("-" * 65)

    print(
        domain_df[
            [
                "domain",
                "n",
                "accuracy",
                "mean_confidence",
                "errors",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Difficulty analysis
    # ---------------------------------------------------------

    difficulty_rows = []

    for difficulty, group in df.groupby("difficulty"):

        row = {
            "difficulty": difficulty,
            "n": len(group),
            "accuracy": group["correct"].mean(),
            "mean_confidence": group["confidence"].mean(),
            "errors": int((group["correct"] == 0).sum()),
        }

        for threshold in THRESHOLDS:

            metrics = threshold_metrics(
                group,
                threshold,
            )

            row[
                f"coverage_{threshold:.2f}"
            ] = metrics["coverage"]

            row[
                f"risk_{threshold:.2f}"
            ] = metrics["selective_risk"]

        difficulty_rows.append(row)

    difficulty_df = pd.DataFrame(
        difficulty_rows
    )

    difficulty_df.to_csv(
        "experiments/data/e3_difficulty_summary.csv",
        index=False,
    )

    print("\nDIFFICULTY ANALYSIS")
    print("-" * 65)

    print(
        difficulty_df[
            [
                "difficulty",
                "n",
                "accuracy",
                "mean_confidence",
                "errors",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Risk-coverage
    # ---------------------------------------------------------

    rc = risk_coverage(df)

    rc.to_csv(
        "experiments/data/e3_risk_coverage_final.csv",
        index=False,
    )

    aurc = float(
        np.trapezoid(
            rc["selective_risk"],
            rc["coverage"],
        )
    )

    print("\nRISK-COVERAGE")
    print("-" * 65)
    print(f"AURC: {aurc:.6f}")

    # ---------------------------------------------------------
    # Error ranking
    # ---------------------------------------------------------

    print("\nERRORS ORDERED BY CONFIDENCE")
    print("-" * 65)

    errors = df[df["correct"] == 0].sort_values(
        "confidence",
        ascending=False,
    )

    print(
        errors[
            [
                "original_id",
                "domain",
                "difficulty",
                "label",
                "probability",
                "confidence",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save threshold results
    # ---------------------------------------------------------

    threshold_df.to_csv(
        "experiments/data/e3_final_threshold_summary.csv",
        index=False,
    )

    print("\nSaved:")
    print("  experiments/data/e3_domain_summary.csv")
    print("  experiments/data/e3_difficulty_summary.csv")
    print("  experiments/data/e3_risk_coverage_final.csv")
    print("  experiments/data/e3_final_threshold_summary.csv")


if __name__ == "__main__":
    main()
