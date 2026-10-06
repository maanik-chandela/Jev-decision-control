from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# E11 DECISION DECOMPOSITION ANALYSIS
# ============================================================

PROJECT_ROOT = Path.cwd()

RESULTS_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "e11_decision_decomposition_results.csv"
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 10_000
SEED = 42


def bootstrap_ci(values, statistic_fn, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Case-level bootstrap.
    The input values must represent independent underlying cases.
    """
    values = np.asarray(values)

    rng = np.random.default_rng(seed)

    n = len(values)

    samples = rng.integers(
        0,
        n,
        size=(n_bootstrap, n),
    )

    boot_stats = np.array(
        [
            statistic_fn(values[idx])
            for idx in samples
        ]
    )

    return (
        float(np.percentile(boot_stats, 2.5)),
        float(np.percentile(boot_stats, 97.5)),
    )


def mean_ci(values, seed=SEED):
    return bootstrap_ci(
        values,
        lambda x: np.mean(x),
        seed=seed,
    )


def validate_results(df):
    print("\n" + "=" * 70)
    print("E11 RESULT VALIDATION")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"Cases: {df['case_id'].nunique()}")

    conditions = df["condition"].value_counts()

    print("\nConditions:")
    print(conditions)

    assert df["case_id"].nunique() == 60
    assert len(df) == 180

    expected_conditions = {
        "direct": 60,
        "decomposed": 60,
        "batched": 60,
    }

    for condition, expected in expected_conditions.items():
        actual = int((df["condition"] == condition).sum())
        assert actual == expected, (
            f"{condition}: expected {expected}, got {actual}"
        )

    duplicate_count = df.duplicated(
        subset=["case_id", "condition"]
    ).sum()

    assert duplicate_count == 0

    print("\nValidation PASSED.")


def build_case_level(df):
    """
    Convert the 180 condition rows into one row per underlying case.

    This is the correct statistical unit for E11.
    """

    direct = (
        df[df["condition"] == "direct"]
        .copy()
        .set_index("case_id")
    )

    decomposed = (
        df[df["condition"] == "decomposed"]
        .copy()
        .set_index("case_id")
    )

    batched = (
        df[df["condition"] == "batched"]
        .copy()
        .set_index("case_id")
    )

    cases = pd.DataFrame(index=direct.index)

    # Metadata
    cases["domain"] = direct["domain"]
    cases["difficulty"] = direct["difficulty"]
    cases["label"] = direct["label"]

    # Direct
    cases["direct_probability"] = direct["probability"]
    cases["direct_prediction"] = direct["prediction"]
    cases["direct_correct"] = direct["correct"]

    # Decomposition
    cases["decomp_probability"] = decomposed["probability"]
    cases["decomp_prediction"] = decomposed["prediction"]
    cases["decomp_correct"] = decomposed["correct"]

    cases["p_A"] = decomposed["p_A"]
    cases["p_B"] = decomposed["p_B"]

    cases["prediction_A"] = decomposed["prediction_A"]
    cases["prediction_B"] = decomposed["prediction_B"]

    cases["correct_A"] = decomposed["correct_A"]
    cases["correct_B"] = decomposed["correct_B"]

    # Batched
    cases["batch_probability"] = batched["probability"]
    cases["batch_prediction"] = batched["prediction"]
    cases["batch_correct"] = batched["correct"]

    cases["distractor_1_probability"] = (
        batched["distractor_1_probability"]
    )

    cases["distractor_2_probability"] = (
        batched["distractor_2_probability"]
    )

    # --------------------------------------------------------
    # Derived metrics
    # --------------------------------------------------------

    cases["direct_vs_decomp_prediction_change"] = (
        cases["direct_prediction"]
        != cases["decomp_prediction"]
    ).astype(int)

    cases["direct_vs_batch_prediction_change"] = (
        cases["direct_prediction"]
        != cases["batch_prediction"]
    ).astype(int)

    cases["direct_vs_decomp_abs_probability_change"] = (
        cases["direct_probability"]
        - cases["decomp_probability"]
    ).abs()

    cases["direct_vs_batch_abs_probability_change"] = (
        cases["direct_probability"]
        - cases["batch_probability"]
    ).abs()

    cases["decomp_probability_spread"] = (
        cases[["p_A", "p_B"]].max(axis=1)
        - cases[["p_A", "p_B"]].min(axis=1)
    )

    # Probability difference using the conservative minimum
    # component probability.
    cases["decomp_min_probability"] = cases[["p_A", "p_B"]].min(
        axis=1
    )

    cases["direct_vs_decomp_min_abs_difference"] = (
        cases["direct_probability"]
        - cases["decomp_min_probability"]
    ).abs()

    # --------------------------------------------------------
    # Key logical diagnostic:
    #
    # Did decomposition identify both components as true?
    # If yes, compound A AND B is true.
    # --------------------------------------------------------

    cases["both_components_true"] = (
        (cases["prediction_A"] == 1)
        & (cases["prediction_B"] == 1)
    ).astype(int)

    cases["decomp_prediction_matches_AND"] = (
        cases["decomp_prediction"]
        == cases["both_components_true"]
    ).astype(int)

    return cases.reset_index()


def print_accuracy_summary(cases):
    print("\n" + "=" * 70)
    print("E11 ACCURACY SUMMARY")
    print("=" * 70)

    for name, column in [
        ("Direct", "direct_correct"),
        ("Decomposed", "decomp_correct"),
        ("Batched", "batch_correct"),
    ]:
        print(
            f"{name:15s}: "
            f"{cases[column].mean():.4f}"
        )

    print(
        "\nDirect vs decomposed binary changes: "
        f"{cases['direct_vs_decomp_prediction_change'].sum()}/"
        f"{len(cases)} "
        f"({cases['direct_vs_decomp_prediction_change'].mean():.4f})"
    )

    print(
        "Direct vs batched binary changes: "
        f"{cases['direct_vs_batch_prediction_change'].sum()}/"
        f"{len(cases)} "
        f"({cases['direct_vs_batch_prediction_change'].mean():.4f})"
    )

    print(
        "\nDecomposition correctly implements A AND B: "
        f"{cases['decomp_prediction_matches_AND'].mean():.4f}"
    )


def print_probability_summary(cases):
    print("\n" + "=" * 70)
    print("E11 PROBABILITY-LEVEL SUMMARY")
    print("=" * 70)

    metrics = [
        (
            "Direct vs decomposed",
            "direct_vs_decomp_abs_probability_change",
        ),
        (
            "Direct vs batched",
            "direct_vs_batch_abs_probability_change",
        ),
        (
            "A vs B component spread",
            "decomp_probability_spread",
        ),
        (
            "Direct vs min(pA,pB)",
            "direct_vs_decomp_min_abs_difference",
        ),
    ]

    rows = []

    for name, column in metrics:
        values = cases[column].to_numpy()

        ci_low, ci_high = mean_ci(values)

        print(f"\n{name}")
        print(f"  Mean absolute change: {values.mean():.4f}")
        print(f"  Median absolute change: {np.median(values):.4f}")
        print(f"  Maximum absolute change: {values.max():.4f}")
        print(
            f"  Bootstrap 95% CI: "
            f"[{ci_low:.4f}, {ci_high:.4f}]"
        )

        rows.append(
            {
                "comparison": name,
                "mean_absolute_change": values.mean(),
                "median_absolute_change": np.median(values),
                "maximum_absolute_change": values.max(),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "e11_probability_summary.csv",
        index=False,
    )


def print_largest_changes(cases):
    print("\n" + "=" * 70)
    print("LARGEST PROBABILITY CHANGES")
    print("=" * 70)

    cols = [
        "case_id",
        "domain",
        "difficulty",
        "label",
        "direct_probability",
        "decomp_probability",
        "batch_probability",
        "direct_vs_decomp_abs_probability_change",
        "direct_vs_batch_abs_probability_change",
        "direct_vs_decomp_prediction_change",
        "direct_vs_batch_prediction_change",
    ]

    largest_decomp = (
        cases.sort_values(
            "direct_vs_decomp_abs_probability_change",
            ascending=False,
        )
        .head(10)[cols]
    )

    largest_batch = (
        cases.sort_values(
            "direct_vs_batch_abs_probability_change",
            ascending=False,
        )
        .head(10)[cols]
    )

    print("\nTop direct → decomposed changes:")
    print(largest_decomp.to_string(index=False))

    print("\nTop direct → batched changes:")
    print(largest_batch.to_string(index=False))

    largest_decomp.to_csv(
        RESULTS_DIR / "e11_largest_decomposition_changes.csv",
        index=False,
    )

    largest_batch.to_csv(
        RESULTS_DIR / "e11_largest_batch_changes.csv",
        index=False,
    )


def domain_analysis(cases):
    print("\n" + "=" * 70)
    print("DOMAIN ANALYSIS")
    print("=" * 70)

    rows = []

    for domain, group in cases.groupby("domain"):

        rows.append(
            {
                "domain": domain,
                "n_cases": len(group),
                "direct_accuracy": group["direct_correct"].mean(),
                "decomposed_accuracy": group["decomp_correct"].mean(),
                "batched_accuracy": group["batch_correct"].mean(),
                "direct_decomp_abs_change": (
                    group[
                        "direct_vs_decomp_abs_probability_change"
                    ].mean()
                ),
                "direct_batch_abs_change": (
                    group[
                        "direct_vs_batch_abs_probability_change"
                    ].mean()
                ),
                "decomp_decision_change_rate": (
                    group[
                        "direct_vs_decomp_prediction_change"
                    ].mean()
                ),
                "batch_decision_change_rate": (
                    group[
                        "direct_vs_batch_prediction_change"
                    ].mean()
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        RESULTS_DIR / "e11_domain_summary.csv",
        index=False,
    )


def difficulty_analysis(cases):
    print("\n" + "=" * 70)
    print("DIFFICULTY ANALYSIS")
    print("=" * 70)

    rows = []

    for difficulty, group in cases.groupby("difficulty"):

        rows.append(
            {
                "difficulty": difficulty,
                "n_cases": len(group),
                "direct_accuracy": group["direct_correct"].mean(),
                "decomposed_accuracy": group["decomp_correct"].mean(),
                "batched_accuracy": group["batch_correct"].mean(),
                "direct_decomp_abs_change": (
                    group[
                        "direct_vs_decomp_abs_probability_change"
                    ].mean()
                ),
                "direct_batch_abs_change": (
                    group[
                        "direct_vs_batch_abs_probability_change"
                    ].mean()
                ),
                "decomp_decision_change_rate": (
                    group[
                        "direct_vs_decomp_prediction_change"
                    ].mean()
                ),
                "batch_decision_change_rate": (
                    group[
                        "direct_vs_batch_prediction_change"
                    ].mean()
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        RESULTS_DIR / "e11_difficulty_summary.csv",
        index=False,
    )


def bootstrap_decision_change(cases):
    print("\n" + "=" * 70)
    print("BOOTSTRAP INFERENCE")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    n = len(cases)

    indices = rng.integers(
        0,
        n,
        size=(N_BOOTSTRAP, n),
    )

    decomp_changes = cases[
        "direct_vs_decomp_prediction_change"
    ].to_numpy()

    batch_changes = cases[
        "direct_vs_batch_prediction_change"
    ].to_numpy()

    decomp_boot = decomp_changes[indices].mean(axis=1)
    batch_boot = batch_changes[indices].mean(axis=1)

    results = []

    for name, observed, samples in [
        (
            "direct_vs_decomposed_decision_change",
            decomp_changes.mean(),
            decomp_boot,
        ),
        (
            "direct_vs_batched_decision_change",
            batch_changes.mean(),
            batch_boot,
        ),
    ]:

        low, high = np.percentile(
            samples,
            [2.5, 97.5],
        )

        print(
            f"{name}: "
            f"{observed:.4f} "
            f"[{low:.4f}, {high:.4f}]"
        )

        results.append(
            {
                "metric": name,
                "observed": observed,
                "ci_low": low,
                "ci_high": high,
            }
        )

    pd.DataFrame(results).to_csv(
        RESULTS_DIR / "e11_bootstrap_summary.csv",
        index=False,
    )


def save_case_level_results(cases):
    output = (
        RESULTS_DIR
        / "e11_case_level_results.csv"
    )

    cases.to_csv(output, index=False)

    print(
        f"\nCase-level results saved to: {output}"
    )


def main():

    print("=" * 70)
    print("E11 DECISION DECOMPOSITION ANALYSIS")
    print("=" * 70)

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{RESULTS_FILE}"
        )

    df = pd.read_csv(RESULTS_FILE)

    validate_results(df)

    cases = build_case_level(df)

    print_accuracy_summary(cases)

    print_probability_summary(cases)

    print_largest_changes(cases)

    domain_analysis(cases)

    difficulty_analysis(cases)

    bootstrap_decision_change(cases)

    save_case_level_results(cases)

    print("\n" + "=" * 70)
    print("E11 ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()