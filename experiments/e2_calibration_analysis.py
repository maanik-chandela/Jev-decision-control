import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE_DIR
    / "experiments"
    / "data"
    / "e2_calibration_results.csv"
)

RELIABILITY_PATH = (
    BASE_DIR
    / "experiments"
    / "data"
    / "e2_reliability_bins.csv"
)

DOMAIN_PATH = (
    BASE_DIR
    / "experiments"
    / "data"
    / "e2_domain_calibration.csv"
)

DIFFICULTY_PATH = (
    BASE_DIR
    / "experiments"
    / "data"
    / "e2_difficulty_calibration.csv"
)

BOOTSTRAP_PATH = (
    BASE_DIR
    / "experiments"
    / "data"
    / "e2_bootstrap_summary.csv"
)


# ============================================================
# METRICS
# ============================================================

def brier_score(y, p):
    return np.mean((p - y) ** 2)


def log_loss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)

    return -np.mean(
        y * np.log(p)
        + (1 - y) * np.log(1 - p)
    )


def ece(y, p, n_bins=10):
    """
    Expected Calibration Error.

    Bins are fixed in probability space:
    [0.0, 0.1), ..., [0.9, 1.0].
    """

    edges = np.linspace(0, 1, n_bins + 1)

    total_error = 0.0
    rows = []

    for i in range(n_bins):

        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)

        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "count": 0,
                "mean_probability": np.nan,
                "empirical_accuracy": np.nan,
                "calibration_error": np.nan,
            })
            continue

        mean_probability = float(p[mask].mean())
        empirical_accuracy = float(y[mask].mean())

        error = abs(
            mean_probability - empirical_accuracy
        )

        weighted_error = (
            count / len(y)
        ) * error

        total_error += weighted_error

        rows.append({
            "bin": i + 1,
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_probability": mean_probability,
            "empirical_accuracy": empirical_accuracy,
            "calibration_error": error,
        })

    return float(total_error), pd.DataFrame(rows)


def confidence_metrics(y, p):
    """
    Secondary analysis.

    Confidence = probability assigned to the predicted class.

    For a true label:
        confidence = p

    For a false label:
        confidence = 1-p

    Correctness = whether the binary decision was correct.
    """

    prediction = (p >= 0.5).astype(int)
    correct = (prediction == y).astype(int)

    confidence = np.where(
        y == 1,
        p,
        1 - p,
    )

    confidence_ece, _ = ece(
        correct,
        confidence,
        n_bins=10,
    )

    return {
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": float(
            confidence[correct == 1].mean()
        ),
        "mean_confidence_incorrect": float(
            confidence[correct == 0].mean()
        ),
        "confidence_ece": confidence_ece,
    }


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_metrics(
    y,
    p,
    n_bootstrap=10000,
    seed=42,
):
    """
    Case-level bootstrap.

    Each row is one independent E2 case.
    """

    rng = np.random.default_rng(seed)

    n = len(y)

    brier_values = []
    logloss_values = []
    ece_values = []
    accuracy_values = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        y_sample = y[indices]
        p_sample = p[indices]

        brier_values.append(
            brier_score(y_sample, p_sample)
        )

        logloss_values.append(
            log_loss(y_sample, p_sample)
        )

        ece_value, _ = ece(
            y_sample,
            p_sample,
            n_bins=10,
        )

        ece_values.append(ece_value)

        predictions = (p_sample >= 0.5).astype(int)

        accuracy_values.append(
            np.mean(predictions == y_sample)
        )

    def ci(values):
        return (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )

    return {
        "metric": [
            "accuracy",
            "brier_score",
            "log_loss",
            "ece",
        ],
        "estimate": [
            np.mean((p >= 0.5) == y),
	    brier_score(y, p),
            log_loss(y, p),
            ece(y, p)[0],
        ],
        "ci_lower": [
            ci(accuracy_values)[0],
            ci(brier_values)[0],
            ci(logloss_values)[0],
            ci(ece_values)[0],
        ],
        "ci_upper": [
            ci(accuracy_values)[1],
            ci(brier_values)[1],
            ci(logloss_values)[1],
            ci(ece_values)[1],
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("E2 CALIBRATION ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    required = {
        "original_id",
        "domain",
        "difficulty",
        "label",
        "probability",
        "prediction",
        "correct",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(df) != 100:
        raise RuntimeError(
            f"Expected 100 cases, found {len(df)}"
        )

    if df["probability"].isna().any():
        raise RuntimeError(
            "Missing probability values detected."
        )

    y = df["label"].to_numpy(dtype=float)
    p = df["probability"].to_numpy(dtype=float)

    predictions = (p >= 0.5).astype(int)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = np.mean(predictions == y)

    brier = brier_score(y, p)
    ll = log_loss(y, p)

    ece_value, reliability = ece(
        y,
        p,
        n_bins=10,
    )

    confidence = confidence_metrics(y, p)

    print()
    print("OVERALL PERFORMANCE")
    print("-" * 60)
    print(f"Cases: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Brier score: {brier:.6f}")
    print(f"Log loss: {ll:.6f}")
    print(f"ECE: {ece_value:.6f}")

    print()
    print("PROBABILITY")
    print("-" * 60)
    print(f"Mean probability: {p.mean():.4f}")
    print(f"Median probability: {np.median(p):.4f}")
    print(f"Minimum probability: {p.min():.4f}")
    print(f"Maximum probability: {p.max():.4f}")

    print()
    print("CONFIDENCE")
    print("-" * 60)
    print(
        f"Mean confidence: "
        f"{confidence['mean_confidence']:.4f}"
    )
    print(
        f"Mean confidence — correct: "
        f"{confidence['mean_confidence_correct']:.4f}"
    )
    print(
        f"Mean confidence — incorrect: "
        f"{confidence['mean_confidence_incorrect']:.4f}"
    )
    print(
        f"Confidence ECE: "
        f"{confidence['confidence_ece']:.6f}"
    )

    # --------------------------------------------------------
    # Reliability table
    # --------------------------------------------------------

    print()
    print("RELIABILITY BINS")
    print("-" * 60)

    print(
        reliability[
            [
                "bin",
                "lower",
                "upper",
                "count",
                "mean_probability",
                "empirical_accuracy",
                "calibration_error",
            ]
        ].to_string(index=False)
    )

    reliability.to_csv(
        RELIABILITY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Domain calibration
    # --------------------------------------------------------

    domain_rows = []

    for domain, group in df.groupby("domain"):

        yd = group["label"].to_numpy(dtype=float)
        pd_ = group["probability"].to_numpy(dtype=float)

        domain_ece, _ = ece(
            yd,
            pd_,
            n_bins=10,
        )

        domain_rows.append({
            "domain": domain,
            "n": len(group),
            "accuracy": np.mean(
                (pd_ >= 0.5) == yd
            ),
            "mean_probability": pd_.mean(),
            "brier_score": brier_score(yd, pd_),
            "log_loss": log_loss(yd, pd_),
            "ece": domain_ece,
        })

    domain_df = pd.DataFrame(domain_rows)

    print()
    print("DOMAIN CALIBRATION")
    print("-" * 60)
    print(domain_df.to_string(index=False))

    domain_df.to_csv(
        DOMAIN_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Difficulty analysis
    # --------------------------------------------------------

    difficulty_rows = []

    for difficulty, group in df.groupby("difficulty"):

        yd = group["label"].to_numpy(dtype=float)
        pd_ = group["probability"].to_numpy(dtype=float)

        difficulty_ece, _ = ece(
            yd,
            pd_,
            n_bins=10,
        )

        difficulty_rows.append({
            "difficulty": difficulty,
            "n": len(group),
            "accuracy": np.mean(
                (pd_ >= 0.5) == yd
            ),
            "mean_probability": pd_.mean(),
            "brier_score": brier_score(yd, pd_),
            "log_loss": log_loss(yd, pd_),
            "ece": difficulty_ece,
        })

    difficulty_df = pd.DataFrame(
        difficulty_rows
    )

    print()
    print("DIFFICULTY ANALYSIS")
    print("-" * 60)
    print(difficulty_df.to_string(index=False))

    difficulty_df.to_csv(
        DIFFICULTY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Incorrect cases
    # --------------------------------------------------------

    incorrect = df[df["correct"] == 0]

    print()
    print("INCORRECT CASES")
    print("-" * 60)

    if len(incorrect) == 0:
        print("None")
    else:
        print(
            incorrect[
                [
                    "original_id",
                    "domain",
                    "difficulty",
                    "label",
                    "probability",
                    "prediction",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    print()
    print("BOOTSTRAP — 10,000 CASE-LEVEL RESAMPLES")
    print("-" * 60)

    bootstrap = bootstrap_metrics(
        y,
        p,
        n_bootstrap=10000,
        seed=42,
    )

    bootstrap_df = pd.DataFrame(bootstrap)

    print(bootstrap_df.to_string(index=False))

    bootstrap_df.to_csv(
        BOOTSTRAP_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("E2 ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"Reliability bins: {RELIABILITY_PATH}")
    print(f"Domain analysis: {DOMAIN_PATH}")
    print(f"Difficulty analysis: {DIFFICULTY_PATH}")
    print(f"Bootstrap summary: {BOOTSTRAP_PATH}")


if __name__ == "__main__":
    main()
