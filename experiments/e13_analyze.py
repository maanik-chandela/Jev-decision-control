import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = "/Users/maanikchandela/JEV-Research/jev-decision-control"

INDEPENDENT_FILE = os.path.join(
    BASE_DIR,
    "experiments/data/e13_independent_results.csv"
)

BATCH_FILE = os.path.join(
    BASE_DIR,
    "experiments/data/e13_batch_results.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "experiments/results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Metrics
# ============================================================

def brier_score(y_true, p):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    return np.mean((p - y_true) ** 2)


def log_loss(y_true, p, eps=1e-15):
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p), eps, 1 - eps)

    return -np.mean(
        y_true * np.log(p)
        + (1 - y_true) * np.log(1 - p)
    )


def ece_score(y_true, p, n_bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(p)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= bins[i]) & (p <= bins[i + 1])
        else:
            mask = (p >= bins[i]) & (p < bins[i + 1])

        if np.sum(mask) == 0:
            continue

        accuracy = np.mean(y_true[mask])
        confidence = np.mean(p[mask])
        weight = np.mean(mask)

        ece += weight * abs(accuracy - confidence)

    return ece


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_mean(values, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    estimates = []

    for _ in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )
        estimates.append(np.mean(sample))

    estimates = np.asarray(estimates)

    return (
        float(np.mean(values)),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


# ============================================================
# Load data
# ============================================================

ind = pd.read_csv(INDEPENDENT_FILE)
batch = pd.read_csv(BATCH_FILE)

print("=" * 70)
print("E13 DECISION-TYPE SENSITIVITY ANALYSIS")
print("=" * 70)

print("\nIndependent rows:", len(ind))
print("Batch rows:", len(batch))


# ============================================================
# Inspect columns
# ============================================================

print("\nIndependent columns:")
print(ind.columns.tolist())

print("\nBatch columns:")
print(batch.columns.tolist())


# ============================================================
# Normalize representation names
# ============================================================

def normalize_type(x):
    x = str(x).lower()

    if "noul" in x:
        return "Noul"

    if "choice" in x:
        return "Choice"

    if "score" in x:
        return "Score"

    return x


# Try to identify the decision-type column.
possible_type_cols = [
    "decision_type",
    "type",
    "representation",
    "question_type"
]

type_col = None

for col in possible_type_cols:
    if col in ind.columns:
        type_col = col
        break

if type_col is None:
    raise ValueError(
        "Could not find decision-type column. "
        f"Available columns: {ind.columns.tolist()}"
    )

ind["decision_type"] = ind[type_col].apply(normalize_type)


# ============================================================
# Identify probability column
# ============================================================

possible_prob_cols = [
    "probability_true",
    "p_true",
    "probability",
    "p",
    "true_probability"
]

prob_col = None

for col in possible_prob_cols:
    if col in ind.columns:
        prob_col = col
        break

if prob_col is None:
    raise ValueError(
        "Could not find probability column. "
        f"Available columns: {ind.columns.tolist()}"
    )

ind["p_true"] = pd.to_numeric(ind[prob_col])


# ============================================================
# Identify label column
# ============================================================

possible_label_cols = [
    "label",
    "ground_truth",
    "target"
]

label_col = None

for col in possible_label_cols:
    if col in ind.columns:
        label_col = col
        break

if label_col is None:
    raise ValueError(
        "Could not find label column. "
        f"Available columns: {ind.columns.tolist()}"
    )

ind["label"] = pd.to_numeric(ind[label_col]).astype(int)


# ============================================================
# Prediction
# ============================================================

ind["prediction"] = (ind["p_true"] >= 0.5).astype(int)
ind["correct"] = (ind["prediction"] == ind["label"]).astype(int)


# ============================================================
# 1. Accuracy by decision type
# ============================================================

accuracy_summary = (
    ind.groupby("decision_type")
    .agg(
        n=("label", "size"),
        accuracy=("correct", "mean"),
        mean_probability=("p_true", "mean"),
        mean_confidence=(
            "p_true",
            lambda x: np.mean(np.maximum(x, 1 - x))
        )
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("1. ACCURACY BY DECISION TYPE")
print("=" * 70)
print(accuracy_summary.to_string(index=False))


# ============================================================
# 2. Calibration
# ============================================================

calibration_rows = []

for dtype in ["Noul", "Choice", "Score"]:
    subset = ind[ind["decision_type"] == dtype]

    calibration_rows.append({
        "decision_type": dtype,
        "accuracy": subset["correct"].mean(),
        "brier": brier_score(
            subset["label"],
            subset["p_true"]
        ),
        "log_loss": log_loss(
            subset["label"],
            subset["p_true"]
        ),
        "ece": ece_score(
            subset["label"],
            subset["p_true"]
        )
    })

calibration = pd.DataFrame(calibration_rows)

print("\n" + "=" * 70)
print("2. CALIBRATION")
print("=" * 70)
print(calibration.to_string(index=False))


# ============================================================
# 3. Pivot probability values by case
# ============================================================

prob_pivot = (
    ind.pivot_table(
        index="case_id",
        columns="decision_type",
        values="p_true",
        aggfunc="first"
    )
)

label_by_case = (
    ind.groupby("case_id")["label"]
    .first()
)

prob_pivot["label"] = label_by_case


# ============================================================
# 4. Pairwise probability differences
# ============================================================

pairs = [
    ("Noul", "Choice"),
    ("Noul", "Score"),
    ("Choice", "Score")
]

pairwise_rows = []

for a, b in pairs:
    diff = (prob_pivot[a] - prob_pivot[b]).abs()

    mean_diff, low, high = bootstrap_mean(diff.values)

    pairwise_rows.append({
        "comparison": f"{a} vs {b}",
        "mean_abs_difference": mean_diff,
        "ci_lower": low,
        "ci_upper": high,
        "median_abs_difference": diff.median(),
        "max_abs_difference": diff.max(),
        "binary_flips": (
            (prob_pivot[a] >= 0.5)
            !=
            (prob_pivot[b] >= 0.5)
        ).sum()
    })

pairwise = pd.DataFrame(pairwise_rows)

print("\n" + "=" * 70)
print("3. PAIRWISE PROBABILITY DIFFERENCES")
print("=" * 70)
print(pairwise.to_string(index=False))


# ============================================================
# 5. Case-level representation spread
# ============================================================

prob_only = prob_pivot[["Noul", "Choice", "Score"]]

prob_pivot["representation_spread"] = (
    prob_only.max(axis=1)
    -
    prob_only.min(axis=1)
)

spread = prob_pivot["representation_spread"]

mean_spread, spread_low, spread_high = bootstrap_mean(
    spread.values
)

print("\n" + "=" * 70)
print("4. CASE-LEVEL REPRESENTATION SPREAD")
print("=" * 70)

print(f"Mean spread   : {mean_spread:.6f}")
print(f"95% CI        : [{spread_low:.6f}, {spread_high:.6f}]")
print(f"Median spread : {spread.median():.6f}")
print(f"Maximum spread: {spread.max():.6f}")


# ============================================================
# 6. Binary decision flips
# ============================================================

print("\nBinary decision flips:")

for a, b in pairs:
    flips = (
        (prob_pivot[a] >= 0.5)
        !=
        (prob_pivot[b] >= 0.5)
    )

    print(
        f"{a} vs {b}: "
        f"{int(flips.sum())}/{len(flips)}"
    )


# Any representation disagreement
binary_matrix = prob_only >= 0.5

any_disagreement = (
    binary_matrix.max(axis=1)
    !=
    binary_matrix.min(axis=1)
)

print(
    "Any representation disagreement:",
    f"{int(any_disagreement.sum())}/{len(any_disagreement)}"
)


# ============================================================
# 7. Independent vs mixed batch
# ============================================================

# Identify batch probability columns.
batch_prob_cols = [
    "noul_probability_true",
    "choice_probability_true",
    "score_probability_true"
]

available_batch_cols = [
    c for c in batch_prob_cols
    if c in batch.columns
]

print("\n" + "=" * 70)
print("5. MIXED-BATCH RESULTS")
print("=" * 70)

if len(available_batch_cols) == 3:

    batch_prob = batch[available_batch_cols].copy()

    batch_prob.columns = [
        "Noul",
        "Choice",
        "Score"
    ]

    batch_prob.index = batch["case_id"]

    for dtype in ["Noul", "Choice", "Score"]:
        independent_values = (
            prob_pivot[dtype]
            .reindex(batch_prob.index)
        )

        batch_values = batch_prob[dtype]

        diff = (
            independent_values - batch_values
        ).abs()

        mean_diff, low, high = bootstrap_mean(
            diff.dropna().values
        )

        print(
            f"{dtype}: "
            f"mean |independent-batch| = {mean_diff:.6f}, "
            f"95% CI = [{low:.6f}, {high:.6f}], "
            f"median = {diff.median():.6f}, "
            f"max = {diff.max():.6f}"
        )

else:
    print(
        "Could not automatically identify batch probability columns."
    )
    print("Available columns:")
    print(batch.columns.tolist())


# ============================================================
# 8. Difficulty analysis
# ============================================================

print("\n" + "=" * 70)
print("6. REPRESENTATION SPREAD BY DIFFICULTY")
print("=" * 70)

difficulty_map = (
    ind.groupby("case_id")["difficulty"]
    .first()
)

prob_pivot["difficulty"] = difficulty_map

difficulty_summary = (
    prob_pivot
    .groupby("difficulty")["representation_spread"]
    .agg(
        n="size",
        mean="mean",
        median="median",
        max="max"
    )
    .reset_index()
)

print(difficulty_summary.to_string(index=False))


# ============================================================
# 9. Domain analysis
# ============================================================

domain_map = (
    ind.groupby("case_id")["domain"]
    .first()
)

prob_pivot["domain"] = domain_map

domain_summary = (
    prob_pivot
    .groupby("domain")["representation_spread"]
    .agg(
        n="size",
        mean="mean",
        median="median",
        max="max"
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("7. REPRESENTATION SPREAD BY DOMAIN")
print("=" * 70)

print(domain_summary.to_string(index=False))


# ============================================================
# 10. Save outputs
# ============================================================

accuracy_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "e13_accuracy_summary.csv"
    ),
    index=False
)

calibration.to_csv(
    os.path.join(
        RESULTS_DIR,
        "e13_calibration_summary.csv"
    ),
    index=False
)

pairwise.to_csv(
    os.path.join(
        RESULTS_DIR,
        "e13_pairwise_probability_differences.csv"
    ),
    index=False
)

prob_pivot.reset_index().to_csv(
    os.path.join(
        RESULTS_DIR,
        "e13_case_level_representation.csv"
    ),
    index=False
)

difficulty_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "e13_difficulty_summary.csv"
    ),
    index=False
)

domain_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "e13_domain_summary.csv"
    ),
    index=False
)


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 70)
print("E13 ANALYSIS COMPLETE")
print("=" * 70)

print(f"Results saved to: {RESULTS_DIR}")