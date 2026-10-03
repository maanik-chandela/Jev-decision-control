import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "experiments" / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

PILOT_FILE = DATA_DIR / "pilot_results.csv"
E4_FILE = DATA_DIR / "e4_results.csv"

OUTPUT_FILE = RESULTS_DIR / "e4_pilot_analysis.csv"


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pilot = pd.read_csv(PILOT_FILE)
    e4 = pd.read_csv(E4_FILE)

    # Keep only the columns needed from the original pilot.
    pilot_original = pilot[
        [
            "question_id",
            "jev_probability",
            "predicted_label",
        ]
    ].copy()

    pilot_original = pilot_original.rename(
        columns={
            "question_id": "original_id",
            "jev_probability": "original_probability",
            "predicted_label": "original_prediction",
        }
    )

    # Match every counterfactual to its original question.
    merged = e4.merge(
        pilot_original,
        on="original_id",
        how="left",
        validate="many_to_one",
    )

    # Absolute probability change.
    merged["delta_probability"] = (
        merged["jev_probability"]
        - merged["original_probability"]
    ).abs()

    # Did the binary decision change?
    merged["decision_flip"] = (
        merged["predicted_label"]
        != merged["original_prediction"]
    )

    merged["stable"] = ~merged["decision_flip"]

    # Save row-level analysis.
    merged.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("E4 COUNTERFACTUAL STABILITY ANALYSIS")
    print("=" * 60)

    print(f"\nTotal counterfactuals: {len(merged)}")

    print(
        f"Original matches: "
        f"{merged['original_probability'].notna().sum()}/{len(merged)}"
    )

    print("\nOverall results")

    print(
        f"Decision stability rate: "
        f"{merged['stable'].mean():.4f}"
    )

    print(
        f"Decision flip rate: "
        f"{merged['decision_flip'].mean():.4f}"
    )

    print(
        f"Mean |Δp|: "
        f"{merged['delta_probability'].mean():.4f}"
    )

    print(
        f"Median |Δp|: "
        f"{merged['delta_probability'].median():.4f}"
    )

    print(
        f"Maximum |Δp|: "
        f"{merged['delta_probability'].max():.4f}"
    )

    print("\nBy perturbation type")

    perturbation_summary = (
        merged
        .groupby("perturbation_type")
        .agg(
            n=("delta_probability", "size"),
            mean_delta_p=("delta_probability", "mean"),
            median_delta_p=("delta_probability", "median"),
            max_delta_p=("delta_probability", "max"),
            flip_rate=("decision_flip", "mean"),
            stability_rate=("stable", "mean"),
        )
        .reset_index()
    )

    print(perturbation_summary.to_string(index=False))

    print("\nBy original question")

    question_summary = (
        merged
        .groupby("original_id")
        .agg(
            n=("delta_probability", "size"),
            original_probability=("original_probability", "first"),
            mean_delta_p=("delta_probability", "mean"),
            max_delta_p=("delta_probability", "max"),
            flip_rate=("decision_flip", "mean"),
        )
        .reset_index()
    )

    print(question_summary.to_string(index=False))

    print("\nLargest probability changes")

    largest_changes = merged.sort_values(
        "delta_probability",
        ascending=False,
    )[
        [
            "counterfactual_id",
            "original_id",
            "perturbation_type",
            "original_probability",
            "jev_probability",
            "delta_probability",
            "original_prediction",
            "predicted_label",
            "decision_flip",
        ]
    ]

    print(
        largest_changes.head(10).to_string(index=False)
    )

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()