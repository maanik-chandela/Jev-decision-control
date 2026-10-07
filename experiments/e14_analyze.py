from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# E14 — State Attack Robustness Analysis
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "experiments" / "data"
RESULTS_DIR = ROOT / "experiments" / "results"

DATASET_FILE = DATA_DIR / "e14_state_attack_dataset.csv"
RESULTS_FILE = DATA_DIR / "e14_state_attack_results.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 10_000
RNG_SEED = 42

CONDITIONS = [
    "clean",
    "prompt_injection",
    "instruction_override",
    "authority_impersonation",
    "irrelevant_malicious",
    "adversarial_evidence",
]

ATTACK_CONDITIONS = [
    "prompt_injection",
    "instruction_override",
    "authority_impersonation",
    "irrelevant_malicious",
    "adversarial_evidence",
]


# ============================================================
# Helpers
# ============================================================

def mean_ci(values, n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED):
    """
    Bootstrap a mean over independent cases.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = np.mean(sample)

    return (
        float(np.mean(values)),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def bootstrap_proportion(values, n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED):
    """
    Bootstrap a binary proportion.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = np.mean(sample)

    return (
        float(np.mean(values)),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def expected_calibration_error(probabilities, labels, n_bins=10):
    """
    Standard equal-width Expected Calibration Error.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)

    ece = 0.0

    for lower in np.linspace(0.0, 1.0, n_bins + 1)[:-1]:
        upper = lower + 1.0 / n_bins

        if lower == 0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities > lower) & (probabilities <= upper)

        if not np.any(mask):
            continue

        bin_probability = probabilities[mask]
        bin_labels = labels[mask]

        confidence = np.mean(bin_probability)
        accuracy = np.mean(bin_labels)

        ece += np.sum(mask) / len(probabilities) * abs(
            accuracy - confidence
        )

    return float(ece)


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("E14 — STATE ATTACK ROBUSTNESS ANALYSIS")
print("=" * 70)

print("\nLoading files...")

dataset = pd.read_csv(DATASET_FILE)
results = pd.read_csv(RESULTS_FILE)

print(f"Dataset rows:  {len(dataset)}")
print(f"Result rows:   {len(results)}")


# ============================================================
# Normalize / verify columns
# ============================================================

required_result_columns = {
    "case_id",
    "domain",
    "difficulty",
    "label",
    "condition",
    "probability_true",
    "prediction",
    "correct",
}

missing = required_result_columns - set(results.columns)

if missing:
    raise ValueError(
        f"Missing required result columns: {sorted(missing)}"
    )

required_dataset_columns = {
    "case_id",
    "domain",
    "difficulty",
    "label",
}

missing_dataset = required_dataset_columns - set(dataset.columns)

if missing_dataset:
    raise ValueError(
        f"Missing required dataset columns: {sorted(missing_dataset)}"
    )


# ============================================================
# Basic integrity
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC INTEGRITY")
print("=" * 70)

expected_cases = 60
expected_conditions = 6
expected_rows = expected_cases * expected_conditions

assert len(dataset) == expected_cases, (
    f"Expected {expected_cases} dataset cases, got {len(dataset)}"
)

assert len(results) == expected_rows, (
    f"Expected {expected_rows} result rows, got {len(results)}"
)

assert results["case_id"].nunique() == expected_cases

condition_counts = results["condition"].value_counts()

for condition in CONDITIONS:
    count = int(condition_counts.get(condition, 0))

    assert count == expected_cases, (
        f"{condition}: expected {expected_cases}, got {count}"
    )

assert set(results["condition"].unique()) == set(CONDITIONS)

assert results[
    ["case_id", "condition"]
].duplicated().sum() == 0

assert results["probability_true"].notna().all()
assert results["prediction"].notna().all()
assert results["correct"].notna().all()

assert results["probability_true"].between(0, 1).all()

print("Dataset cases:", len(dataset))
print("Result rows:", len(results))
print("Unique cases:", results["case_id"].nunique())
print("\nCondition counts:")
print(condition_counts.sort_index())

print("\nBasic integrity: PASSED")


# ============================================================
# Verify predictions and correctness
# ============================================================

print("\n" + "=" * 70)
print("2. PREDICTION / CORRECTNESS INTEGRITY")
print("=" * 70)

calculated_prediction = (
    results["probability_true"] >= 0.5
).astype(int)

prediction_mismatches = (
    calculated_prediction != results["prediction"].astype(int)
).sum()

calculated_correct = (
    results["prediction"].astype(int)
    == results["label"].astype(int)
).astype(int)

correctness_mismatches = (
    calculated_correct != results["correct"].astype(int)
).sum()

print("Prediction mismatches:", prediction_mismatches)
print("Correctness mismatches:", correctness_mismatches)

assert prediction_mismatches == 0
assert correctness_mismatches == 0

print("Prediction integrity: PASSED")
print("Correctness integrity: PASSED")


# ============================================================
# Verify labels against dataset
# ============================================================

dataset_labels = dataset.set_index("case_id")["label"]

result_labels = results["case_id"].map(dataset_labels)

label_mismatches = (
    result_labels.astype(int)
    != results["label"].astype(int)
).sum()

print("Label mismatches:", label_mismatches)

assert label_mismatches == 0

print("Dataset/result label integrity: PASSED")


# ============================================================
# 3. Accuracy and confidence by condition
# ============================================================

print("\n" + "=" * 70)
print("3. ACCURACY AND CONFIDENCE BY CONDITION")
print("=" * 70)

condition_summary = []

for condition in CONDITIONS:
    df = results[results["condition"] == condition].copy()

    accuracy = df["correct"].mean()
    mean_probability = df["probability_true"].mean()

    confidence = np.where(
        df["probability_true"] >= 0.5,
        df["probability_true"],
        1.0 - df["probability_true"],
    )

    mean_confidence = np.mean(confidence)

    condition_summary.append({
        "condition": condition,
        "n": len(df),
        "accuracy": accuracy,
        "mean_probability": mean_probability,
        "mean_confidence": mean_confidence,
    })

condition_summary = pd.DataFrame(condition_summary)

print(
    condition_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

condition_summary.to_csv(
    RESULTS_DIR / "e14_condition_summary.csv",
    index=False,
)


# ============================================================
# 4. Calibration by condition
# ============================================================

print("\n" + "=" * 70)
print("4. CALIBRATION BY CONDITION")
print("=" * 70)

calibration_rows = []

for condition in CONDITIONS:
    df = results[results["condition"] == condition]

    p = df["probability_true"].to_numpy()
    y = df["label"].to_numpy()

    brier = np.mean((p - y) ** 2)

    eps = 1e-12
    clipped = np.clip(p, eps, 1 - eps)

    log_loss = -np.mean(
        y * np.log(clipped)
        + (1 - y) * np.log(1 - clipped)
    )

    ece = expected_calibration_error(p, y)

    calibration_rows.append({
        "condition": condition,
        "n": len(df),
        "accuracy": df["correct"].mean(),
        "brier": brier,
        "log_loss": log_loss,
        "ece": ece,
    })

calibration_summary = pd.DataFrame(calibration_rows)

print(
    calibration_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

calibration_summary.to_csv(
    RESULTS_DIR / "e14_calibration_summary.csv",
    index=False,
)


# ============================================================
# 5. Pair clean condition with each attack
# ============================================================

print("\n" + "=" * 70)
print("5. PAIRED ATTACK EFFECTS VS CLEAN")
print("=" * 70)

clean = (
    results[results["condition"] == "clean"]
    .set_index("case_id")
)

attack_summary_rows = []

paired_tables = {}

for attack in ATTACK_CONDITIONS:

    attacked = (
        results[results["condition"] == attack]
        .set_index("case_id")
    )

    common_cases = clean.index.intersection(attacked.index)

    assert len(common_cases) == expected_cases

    pair = pd.DataFrame(index=common_cases)

    pair["label"] = clean.loc[common_cases, "label"].astype(int)

    pair["clean_probability"] = clean.loc[
        common_cases, "probability_true"
    ].astype(float)

    pair["attack_probability"] = attacked.loc[
        common_cases, "probability_true"
    ].astype(float)

    pair["probability_delta"] = (
        pair["attack_probability"]
        - pair["clean_probability"]
    )

    pair["abs_probability_delta"] = (
        pair["probability_delta"].abs()
    )

    pair["clean_prediction"] = clean.loc[
        common_cases, "prediction"
    ].astype(int)

    pair["attack_prediction"] = attacked.loc[
        common_cases, "prediction"
    ].astype(int)

    pair["decision_changed"] = (
        pair["clean_prediction"]
        != pair["attack_prediction"]
    ).astype(int)

    pair["clean_correct"] = clean.loc[
        common_cases, "correct"
    ].astype(int)

    pair["attack_correct"] = attacked.loc[
        common_cases, "correct"
    ].astype(int)

    pair["attack_caused_error"] = (
        (pair["attack_correct"] == 0)
        & (pair["clean_correct"] == 1)
    ).astype(int)

    pair["clean_confidence"] = np.where(
        pair["clean_probability"] >= 0.5,
        pair["clean_probability"],
        1.0 - pair["clean_probability"],
    )

    paired_tables[attack] = pair.reset_index().rename(
        columns={"index": "case_id"}
    )

    mean_delta, delta_lo, delta_hi = mean_ci(
        pair["probability_delta"]
    )

    mean_abs, abs_lo, abs_hi = mean_ci(
        pair["abs_probability_delta"]
    )

    flip_rate, flip_lo, flip_hi = bootstrap_proportion(
        pair["decision_changed"]
    )

    attack_error_rate, error_lo, error_hi = bootstrap_proportion(
        1 - pair["attack_correct"]
    )

    clean_error_rate = 1 - pair["clean_correct"].mean()

    attack_summary_rows.append({
        "attack_condition": attack,
        "n_cases": len(pair),
        "clean_accuracy": pair["clean_correct"].mean(),
        "attack_accuracy": pair["attack_correct"].mean(),
        "clean_error_rate": clean_error_rate,
        "attack_error_rate": attack_error_rate,
        "attack_error_ci_low": error_lo,
        "attack_error_ci_high": error_hi,
        "mean_delta_p": mean_delta,
        "delta_ci_low": delta_lo,
        "delta_ci_high": delta_hi,
        "mean_abs_delta_p": mean_abs,
        "abs_delta_ci_low": abs_lo,
        "abs_delta_ci_high": abs_hi,
        "median_abs_delta_p": pair["abs_probability_delta"].median(),
        "max_abs_delta_p": pair["abs_probability_delta"].max(),
        "decision_changes": pair["decision_changed"].sum(),
        "decision_change_rate": flip_rate,
        "decision_change_ci_low": flip_lo,
        "decision_change_ci_high": flip_hi,
        "attack_caused_errors": pair["attack_caused_error"].sum(),
    })


attack_summary = pd.DataFrame(attack_summary_rows)

print(
    attack_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

attack_summary.to_csv(
    RESULTS_DIR / "e14_attack_summary.csv",
    index=False,
)


# ============================================================
# 6. Identify all decision changes
# ============================================================

print("\n" + "=" * 70)
print("6. DECISION CHANGES VS CLEAN")
print("=" * 70)

all_changes = []

for attack in ATTACK_CONDITIONS:

    pair = paired_tables[attack].copy()

    changed = pair[pair["decision_changed"] == 1].copy()

    if len(changed) > 0:
        changed["attack_condition"] = attack
        all_changes.append(changed)

if all_changes:
    decision_changes = pd.concat(
        all_changes,
        ignore_index=True,
    )

    decision_changes = decision_changes[
        [
            "case_id",
            "attack_condition",
            "label",
            "clean_probability",
            "attack_probability",
            "probability_delta",
            "clean_prediction",
            "attack_prediction",
            "clean_correct",
            "attack_correct",
        ]
    ]

    print(
        decision_changes.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    decision_changes.to_csv(
        RESULTS_DIR / "e14_decision_changes.csv",
        index=False,
    )

else:
    decision_changes = pd.DataFrame()

    print("No decision changes observed.")


# ============================================================
# 7. Domain-level attack sensitivity
# ============================================================

print("\n" + "=" * 70)
print("7. DOMAIN-LEVEL ATTACK SENSITIVITY")
print("=" * 70)

domain_rows = []

for attack in ATTACK_CONDITIONS:

    pair = paired_tables[attack].copy()

    domain_map = (
        results[results["condition"] == "clean"]
        [["case_id", "domain", "difficulty"]]
        .drop_duplicates("case_id")
        .set_index("case_id")
    )

    pair = pair.set_index("case_id").join(domain_map)

    for domain, group in pair.groupby("domain"):

        domain_rows.append({
            "attack_condition": attack,
            "domain": domain,
            "n": len(group),
            "accuracy": group["attack_correct"].mean(),
            "mean_abs_delta_p": group[
                "abs_probability_delta"
            ].mean(),
            "median_abs_delta_p": group[
                "abs_probability_delta"
            ].median(),
            "max_abs_delta_p": group[
                "abs_probability_delta"
            ].max(),
            "decision_changes": group[
                "decision_changed"
            ].sum(),
        })

domain_summary = pd.DataFrame(domain_rows)

print(
    domain_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

domain_summary.to_csv(
    RESULTS_DIR / "e14_domain_summary.csv",
    index=False,
)


# ============================================================
# 8. Difficulty-level attack sensitivity
# ============================================================

print("\n" + "=" * 70)
print("8. DIFFICULTY-LEVEL ATTACK SENSITIVITY")
print("=" * 70)

difficulty_rows = []

for attack in ATTACK_CONDITIONS:

    pair = paired_tables[attack].copy()

    difficulty_map = (
        results[results["condition"] == "clean"]
        [["case_id", "difficulty"]]
        .drop_duplicates("case_id")
        .set_index("case_id")
    )

    pair = pair.set_index("case_id").join(difficulty_map)

    for difficulty, group in pair.groupby("difficulty"):

        difficulty_rows.append({
            "attack_condition": attack,
            "difficulty": difficulty,
            "n": len(group),
            "accuracy": group["attack_correct"].mean(),
            "mean_abs_delta_p": group[
                "abs_probability_delta"
            ].mean(),
            "median_abs_delta_p": group[
                "abs_probability_delta"
            ].median(),
            "max_abs_delta_p": group[
                "abs_probability_delta"
            ].max(),
            "decision_changes": group[
                "decision_changed"
            ].sum(),
        })

difficulty_summary = pd.DataFrame(difficulty_rows)

print(
    difficulty_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

difficulty_summary.to_csv(
    RESULTS_DIR / "e14_difficulty_summary.csv",
    index=False,
)


# ============================================================
# 9. Clean-confidence stratification
# ============================================================

print("\n" + "=" * 70)
print("9. SENSITIVITY BY CLEAN CONFIDENCE")
print("=" * 70)

clean_info = (
    results[results["condition"] == "clean"]
    [
        [
            "case_id",
            "probability_true",
            "label",
            "correct",
        ]
    ]
    .copy()
)

clean_info["clean_confidence"] = np.where(
    clean_info["probability_true"] >= 0.5,
    clean_info["probability_true"],
    1.0 - clean_info["probability_true"],
)

def confidence_bin(conf):
    if conf < 0.75:
        return "low/moderate (<0.75)"
    elif conf < 0.90:
        return "high (0.75–0.89)"
    else:
        return "very_high (>=0.90)"


clean_info["confidence_group"] = (
    clean_info["clean_confidence"]
    .apply(confidence_bin)
)

confidence_rows = []

for attack in ATTACK_CONDITIONS:

    pair = paired_tables[attack].copy()

    pair = pair.merge(
        clean_info[
            [
                "case_id",
                "clean_confidence",
                "confidence_group",
            ]
        ],
        on="case_id",
        how="left",
    )

    for group_name, group in pair.groupby(
        "confidence_group",
        sort=False,
    ):

        confidence_rows.append({
            "attack_condition": attack,
            "confidence_group": group_name,
            "n": len(group),
            "mean_abs_delta_p": group[
                "abs_probability_delta"
            ].mean(),
            "median_abs_delta_p": group[
                "abs_probability_delta"
            ].median(),
            "max_abs_delta_p": group[
                "abs_probability_delta"
            ].max(),
            "decision_changes": group[
                "decision_changed"
            ].sum(),
            "attack_accuracy": group[
                "attack_correct"
            ].mean(),
        })

confidence_summary = pd.DataFrame(confidence_rows)

print(
    confidence_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

confidence_summary.to_csv(
    RESULTS_DIR / "e14_confidence_summary.csv",
    index=False,
)


# ============================================================
# 10. All paired results
# ============================================================

all_pairs = []

for attack in ATTACK_CONDITIONS:

    pair = paired_tables[attack].copy()
    pair["attack_condition"] = attack

    all_pairs.append(pair)

all_pairs = pd.concat(
    all_pairs,
    ignore_index=True,
)

all_pairs.to_csv(
    RESULTS_DIR / "e14_paired_case_results.csv",
    index=False,
)


# ============================================================
# 11. Highlight largest probability shifts
# ============================================================

print("\n" + "=" * 70)
print("10. LARGEST PROBABILITY SHIFTS")
print("=" * 70)

largest = (
    all_pairs
    .sort_values(
        "abs_probability_delta",
        ascending=False,
    )
    .head(20)
)

print(
    largest[
        [
            "case_id",
            "attack_condition",
            "label",
            "clean_probability",
            "attack_probability",
            "probability_delta",
            "abs_probability_delta",
            "decision_changed",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

largest[
    [
        "case_id",
        "attack_condition",
        "label",
        "clean_probability",
        "attack_probability",
        "probability_delta",
        "abs_probability_delta",
        "decision_changed",
    ]
].to_csv(
    RESULTS_DIR / "e14_largest_probability_shifts.csv",
    index=False,
)


# ============================================================
# 12. Headline numbers
# ============================================================

print("\n" + "=" * 70)
print("11. E14 HEADLINE RESULTS")
print("=" * 70)

for _, row in attack_summary.iterrows():

    print(
        f"{row['attack_condition']:24s} "
        f"accuracy={row['attack_accuracy']:.4f}  "
        f"mean|Δp|={row['mean_abs_delta_p']:.4f}  "
        f"median|Δp|={row['median_abs_delta_p']:.4f}  "
        f"max|Δp|={row['max_abs_delta_p']:.4f}  "
        f"decision_changes={int(row['decision_changes'])}/60"
    )


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 70)
print("E14 ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved files:")

for path in sorted(RESULTS_DIR.glob("e14_*.csv")):
    print(" ", path.relative_to(ROOT))

print("\nNo API calls were made by this analysis.")
print("Analysis uses the recorded E14 results only.")