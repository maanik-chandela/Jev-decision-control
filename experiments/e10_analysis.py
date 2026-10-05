import random
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# E10 COMPUTATION ALLOCATION ANALYSIS
# ============================================================
#
# Policies:
#
# 1. Always-Base
#       Always use Qwen3:8B.
#
# 2. Always-Strong
#       Always use the direct JEV answer.
#
# 3. JEV-Router
#       JEV estimates P(Stage-1 answer is correct).
#       If probability >= threshold:
#           accept Qwen answer
#       Else:
#           escalate to JEV strong answer.
#
# 4. Random-Router
#       Escalate exactly the same number of cases as the
#       JEV-Router, but choose those cases randomly.
#
# Bootstrap:
#       Case-level resampling over the 150 independent cases.
#
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "experiments" / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


DATASET_FILE = (
    DATA_DIR / "e10_computation_allocation_dataset.csv"
)

BASE_FILE = (
    DATA_DIR / "e10_base_results.csv"
)

CONTROLLER_FILE = (
    DATA_DIR / "e10_jev_controller_results.csv"
)

STRONG_FILE = (
    DATA_DIR / "e10_jev_strong_results.csv"
)


# TypeSafe JEV pricing:
# $42 per billion input tokens.
JEV_INPUT_COST_PER_BILLION = 42.0

BOOTSTRAP_REPS = 10_000

RANDOM_SEED = 20261005

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]


# ============================================================
# HELPERS
# ============================================================

def jev_input_cost(input_tokens):
    """
    TypeSafe JEV input cost.
    Output tokens are treated as free.
    """

    return (
        float(input_tokens)
        / 1_000_000_000
        * JEV_INPUT_COST_PER_BILLION
    )


def bootstrap_ci(values, statistic=np.mean, reps=BOOTSTRAP_REPS):
    """
    Case-level nonparametric bootstrap 95% CI.
    """

    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(RANDOM_SEED)

    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    samples = rng.integers(
        0,
        n,
        size=(reps, n),
    )

    boot_values = statistic(
        values[samples],
        axis=1,
    )

    estimate = statistic(values)

    lower = np.percentile(
        boot_values,
        2.5,
    )

    upper = np.percentile(
        boot_values,
        97.5,
    )

    return estimate, lower, upper


def bootstrap_difference(
    values_a,
    values_b,
    reps=BOOTSTRAP_REPS,
):
    """
    Paired case-level bootstrap CI for mean(A-B).
    """

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    assert len(values_a) == len(values_b)

    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    n = len(values_a)

    samples = rng.integers(
        0,
        n,
        size=(reps, n),
    )

    differences = (
        values_a[samples]
        - values_b[samples]
    ).mean(axis=1)

    observed = (
        values_a - values_b
    ).mean()

    lower = np.percentile(
        differences,
        2.5,
    )

    upper = np.percentile(
        differences,
        97.5,
    )

    return observed, lower, upper


def accuracy_ci(values):
    return bootstrap_ci(
        values,
        statistic=np.mean,
    )


# ============================================================
# LOAD FILES
# ============================================================

dataset = pd.read_csv(DATASET_FILE)
base = pd.read_csv(BASE_FILE)
controller = pd.read_csv(CONTROLLER_FILE)
strong = pd.read_csv(STRONG_FILE)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_dataset = {
    "case_id",
    "domain",
    "difficulty",
    "question",
    "label",
}

required_base = {
    "case_id",
    "prediction",
    "correct",
}

required_controller = {
    "case_id",
    "jev_probability_correct",
    "jev_prediction_correct",
    "input_tokens",
    "output_tokens",
}

required_strong = {
    "case_id",
    "jev_prediction",
    "jev_correct",
    "input_tokens",
    "output_tokens",
}


for name, df, required in [
    ("dataset", dataset, required_dataset),
    ("base", base, required_base),
    ("controller", controller, required_controller),
    ("strong", strong, required_strong),
]:

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{name} missing columns: {sorted(missing)}"
        )


# ============================================================
# NORMALIZE IDs
# ============================================================

for df in [
    dataset,
    base,
    controller,
    strong,
]:

    df["case_id"] = (
        df["case_id"]
        .astype(str)
    )


# ============================================================
# REMOVE ANY STALE FAILED STRONG ROWS
# ============================================================

if "error" in strong.columns:

    strong = strong[
        strong["error"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    ].copy()


# One successful strong result per case.
strong = strong.drop_duplicates(
    subset=["case_id"],
    keep="last",
)


# ============================================================
# MERGE
# ============================================================

df = dataset[
    [
        "case_id",
        "domain",
        "difficulty",
        "question",
        "label",
    ]
].copy()


base_small = base[
    [
        "case_id",
        "prediction",
        "correct",
    ]
].rename(
    columns={
        "prediction": "stage1_prediction",
        "correct": "stage1_correct",
    }
)


controller_small = controller[
    [
        "case_id",
        "jev_probability_correct",
        "jev_prediction_correct",
        "input_tokens",
        "output_tokens",
    ]
].rename(
    columns={
        "input_tokens": "controller_input_tokens",
        "output_tokens": "controller_output_tokens",
    }
)


strong_small = strong[
    [
        "case_id",
        "jev_prediction",
        "jev_correct",
        "input_tokens",
        "output_tokens",
    ]
].rename(
    columns={
        "input_tokens": "strong_input_tokens",
        "output_tokens": "strong_output_tokens",
    }
)


df = df.merge(
    base_small,
    on="case_id",
    how="inner",
)

df = df.merge(
    controller_small,
    on="case_id",
    how="inner",
)

df = df.merge(
    strong_small,
    on="case_id",
    how="inner",
)


# ============================================================
# FINAL DATA VALIDATION
# ============================================================

if len(df) != 150:
    raise RuntimeError(
        f"Expected 150 merged cases, got {len(df)}"
    )


if df["case_id"].nunique() != 150:
    raise RuntimeError(
        "Case IDs are not unique."
    )


if df["label"].value_counts().to_dict() != {
    0: 75,
    1: 75,
}:
    raise RuntimeError(
        "Unexpected label balance."
    )


if not df["stage1_correct"].isin(
    [0, 1, True, False]
).all():

    raise RuntimeError(
        "Invalid Stage-1 correctness values."
    )


if not df["jev_correct"].isin(
    [0, 1, True, False]
).all():

    raise RuntimeError(
        "Invalid strong-model correctness values."
    )


df["stage1_correct"] = (
    df["stage1_correct"]
    .astype(int)
)

df["jev_correct"] = (
    df["jev_correct"]
    .astype(int)
)

df["jev_probability_correct"] = (
    pd.to_numeric(
        df["jev_probability_correct"],
        errors="raise",
    )
)


# ============================================================
# TOKEN / COST CALCULATIONS
# ============================================================

df["controller_input_tokens"] = (
    pd.to_numeric(
        df["controller_input_tokens"],
        errors="coerce",
    )
    .fillna(0)
)

df["controller_output_tokens"] = (
    pd.to_numeric(
        df["controller_output_tokens"],
        errors="coerce",
    )
    .fillna(0)
)

df["strong_input_tokens"] = (
    pd.to_numeric(
        df["strong_input_tokens"],
        errors="coerce",
    )
    .fillna(0)
)

df["strong_output_tokens"] = (
    pd.to_numeric(
        df["strong_output_tokens"],
        errors="coerce",
    )
    .fillna(0)
)


df["controller_cost_usd"] = (
    df["controller_input_tokens"]
    .apply(jev_input_cost)
)

df["strong_cost_usd"] = (
    df["strong_input_tokens"]
    .apply(jev_input_cost)
)


TOTAL_CONTROLLER_COST = (
    df["controller_cost_usd"].sum()
)

TOTAL_STRONG_COST = (
    df["strong_cost_usd"].sum()
)


# ============================================================
# BASELINE POLICIES
# ============================================================

always_base_accuracy = (
    df["stage1_correct"]
    .mean()
)

always_strong_accuracy = (
    df["jev_correct"]
    .mean()
)


# ============================================================
# JEV CONTROLLER CALIBRATION
# ============================================================

y = df["stage1_correct"].to_numpy(
    dtype=float
)

p = df["jev_probability_correct"].to_numpy(
    dtype=float
)


# Brier score
brier = np.mean(
    (p - y) ** 2
)


# Log loss
eps = 1e-15

p_clipped = np.clip(
    p,
    eps,
    1 - eps,
)

log_loss = -np.mean(
    y * np.log(p_clipped)
    + (1 - y)
    * np.log(1 - p_clipped)
)


# ECE
bins = np.linspace(
    0,
    1,
    11,
)

ece = 0.0

for i in range(10):

    if i == 9:

        mask = (
            (p >= bins[i])
            & (p <= bins[i + 1])
        )

    else:

        mask = (
            (p >= bins[i])
            & (p < bins[i + 1])
        )

    if mask.sum() == 0:
        continue

    confidence = p[mask].mean()
    accuracy = y[mask].mean()

    ece += (
        mask.sum()
        / len(y)
        * abs(
            confidence
            - accuracy
        )
    )


controller_calibration = pd.DataFrame(
    [
        {
            "n_cases": len(df),
            "brier_score": brier,
            "log_loss": log_loss,
            "ece": ece,
            "mean_probability": p.mean(),
            "mean_correct_probability": p[y == 1].mean(),
            "mean_incorrect_probability": (
                p[y == 0].mean()
                if (y == 0).sum() > 0
                else np.nan
            ),
        }
    ]
)


# ============================================================
# POLICY EVALUATION
# ============================================================

policy_rows = []
threshold_rows = []
case_level_rows = []
random_bootstrap_rows = []


# ------------------------------------------------------------
# Always-Base
# ------------------------------------------------------------

base_ci = accuracy_ci(
    df["stage1_correct"].to_numpy()
)

policy_rows.append(
    {
        "policy": "Always-Base",
        "threshold": np.nan,
        "accuracy": always_base_accuracy,
        "accuracy_ci_low": base_ci[1],
        "accuracy_ci_high": base_ci[2],
        "escalation_rate": 0.0,
        "escalated_cases": 0,
        "controller_calls": 0,
        "strong_calls": 0,
        "total_jev_calls": 0,
        "jev_cost_usd": 0.0,
    }
)


# ------------------------------------------------------------
# Always-Strong
# ------------------------------------------------------------

strong_ci = accuracy_ci(
    df["jev_correct"].to_numpy()
)

policy_rows.append(
    {
        "policy": "Always-Strong",
        "threshold": np.nan,
        "accuracy": always_strong_accuracy,
        "accuracy_ci_low": strong_ci[1],
        "accuracy_ci_high": strong_ci[2],
        "escalation_rate": 1.0,
        "escalated_cases": len(df),
        "controller_calls": 0,
        "strong_calls": len(df),
        "total_jev_calls": len(df),
        "jev_cost_usd": TOTAL_STRONG_COST,
    }
)


# ============================================================
# JEV ROUTER + MATCHED RANDOM ROUTER
# ============================================================

for threshold in THRESHOLDS:

    # --------------------------------------------------------
    # JEV routing decision
    # --------------------------------------------------------

    escalate = (
        df["jev_probability_correct"]
        < threshold
    )

    df["jev_escalate"] = (
        escalate.astype(int)
    )

    n_escalated = int(
        escalate.sum()
    )

    escalation_rate = (
        n_escalated / len(df)
    )


    # --------------------------------------------------------
    # JEV Router final prediction
    # --------------------------------------------------------

    router_correct = np.where(
        escalate,
        df["jev_correct"].to_numpy(),
        df["stage1_correct"].to_numpy(),
    )

    router_correct = (
        router_correct.astype(int)
    )

    router_accuracy = (
        router_correct.mean()
    )


    # --------------------------------------------------------
    # Cost
    #
    # Controller is called on all 150 cases.
    # Strong model is called only on escalated cases.
    # --------------------------------------------------------

    controller_calls = len(df)

    strong_calls = n_escalated

    controller_cost = (
        df["controller_cost_usd"]
        .sum()
    )

    strong_cost = (
        df.loc[
            escalate,
            "strong_cost_usd",
        ]
        .sum()
    )

    total_cost = (
        controller_cost
        + strong_cost
    )


    # --------------------------------------------------------
    # JEV Router bootstrap
    # --------------------------------------------------------

    router_estimate, router_low, router_high = (
        accuracy_ci(router_correct)
    )


    # --------------------------------------------------------
    # Random Router
    #
    # For every bootstrap replicate:
    # randomly select exactly n_escalated cases.
    #
    # This creates a matched random-routing baseline.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
        + int(threshold * 1000)
    )

    random_accuracies = []

    random_case_correct = []

    for rep in range(
        BOOTSTRAP_REPS
    ):

        # Case-level bootstrap sample
        indices = rng.integers(
            0,
            len(df),
            size=len(df),
        )

        boot_stage1 = (
            df["stage1_correct"]
            .to_numpy()[indices]
        )

        boot_strong = (
            df["jev_correct"]
            .to_numpy()[indices]
        )

        # Randomly choose exactly the
        # matched number of escalations.
        chosen = np.zeros(
            len(df),
            dtype=bool,
        )

        if n_escalated > 0:

            selected = rng.choice(
                len(df),
                size=n_escalated,
                replace=False,
            )

            chosen[selected] = True

        boot_final = np.where(
            chosen,
            boot_strong,
            boot_stage1,
        )

        random_accuracies.append(
            boot_final.mean()
        )

        if rep < 1000:

            random_case_correct.append(
                boot_final
            )


    random_accuracies = np.asarray(
        random_accuracies
    )

    random_accuracy = (
        np.mean(
            np.where(
                rng.choice(
                    len(df),
                    size=n_escalated,
                    replace=False,
                )
                if n_escalated > 0
                else np.array([], dtype=int),
                0,
                0,
            )
        )
        if False
        else np.nan
    )

    # Calculate observed random-router
    # expectation analytically.
    #
    # Random escalation means the expected
    # accuracy is the weighted average of
    # base and strong accuracy.
    random_expected_accuracy = (
        (
            (len(df) - n_escalated)
            * always_base_accuracy
            + n_escalated
            * always_strong_accuracy
        )
        / len(df)
    )

    random_ci_low = np.percentile(
        random_accuracies,
        2.5,
    )

    random_ci_high = np.percentile(
        random_accuracies,
        97.5,
    )


    # --------------------------------------------------------
    # Matched random router difference
    # --------------------------------------------------------

    improvement_over_random = (
        router_accuracy
        - random_expected_accuracy
    )


    # --------------------------------------------------------
    # Router cost relative to Always-Base
    # --------------------------------------------------------

    cost_per_correct = (
        total_cost / router_correct.sum()
        if router_correct.sum() > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Policy row
    # --------------------------------------------------------

    policy_rows.append(
        {
            "policy": "JEV-Router",
            "threshold": threshold,
            "accuracy": router_accuracy,
            "accuracy_ci_low": router_low,
            "accuracy_ci_high": router_high,
            "escalation_rate": escalation_rate,
            "escalated_cases": n_escalated,
            "controller_calls": controller_calls,
            "strong_calls": strong_calls,
            "total_jev_calls": (
                controller_calls
                + strong_calls
            ),
            "jev_cost_usd": total_cost,
            "cost_per_correct_usd": cost_per_correct,
        }
    )


    # --------------------------------------------------------
    # Threshold analysis row
    # --------------------------------------------------------

    threshold_rows.append(
        {
            "threshold": threshold,
            "escalated_cases": n_escalated,
            "escalation_rate": escalation_rate,

            "jev_router_accuracy": router_accuracy,
            "jev_router_ci_low": router_low,
            "jev_router_ci_high": router_high,

            "random_router_expected_accuracy": (
                random_expected_accuracy
            ),

            "random_router_ci_low": (
                random_ci_low
            ),

            "random_router_ci_high": (
                random_ci_high
            ),

            "jev_router_minus_random": (
                improvement_over_random
            ),

            "controller_calls": controller_calls,
            "strong_calls": strong_calls,
            "total_jev_calls": (
                controller_calls
                + strong_calls
            ),

            "controller_cost_usd": (
                controller_cost
            ),

            "strong_cost_usd": (
                strong_cost
            ),

            "total_jev_cost_usd": (
                total_cost
            ),
        }
    )


    # --------------------------------------------------------
    # Case-level results
    # --------------------------------------------------------

    for i, row in df.iterrows():

        case_level_rows.append(
            {
                "case_id": row["case_id"],
                "domain": row["domain"],
                "difficulty": row["difficulty"],
                "label": row["label"],

                "stage1_correct": row[
                    "stage1_correct"
                ],

                "jev_probability_correct": row[
                    "jev_probability_correct"
                ],

                "strong_correct": row[
                    "jev_correct"
                ],

                "threshold": threshold,

                "jev_escalate": int(
                    escalate.iloc[i]
                ),

                "jev_router_correct": int(
                    router_correct[i]
                ),

                "random_router_expected_accuracy": (
                    random_expected_accuracy
                ),
            }
        )


# ============================================================
# SAVE RESULTS
# ============================================================

policy_summary = pd.DataFrame(
    policy_rows
)

threshold_analysis = pd.DataFrame(
    threshold_rows
)

case_level_results = pd.DataFrame(
    case_level_rows
)

controller_calibration.to_csv(
    RESULTS_DIR
    / "e10_controller_calibration.csv",
    index=False,
)

policy_summary.to_csv(
    RESULTS_DIR
    / "e10_policy_summary.csv",
    index=False,
)

threshold_analysis.to_csv(
    RESULTS_DIR
    / "e10_threshold_analysis.csv",
    index=False,
)

case_level_results.to_csv(
    RESULTS_DIR
    / "e10_case_level_results.csv",
    index=False,
)


# ============================================================
# BOOTSTRAP SUMMARY
# ============================================================

bootstrap_summary_rows = []


for threshold in THRESHOLDS:

    subset = case_level_results[
        case_level_results["threshold"]
        == threshold
    ]

    router_values = (
        subset["jev_router_correct"]
        .to_numpy()
    )

    base_values = (
        subset["stage1_correct"]
        .to_numpy()
    )

    strong_values = (
        subset["strong_correct"]
        .to_numpy()
    )


    router_est, router_low, router_high = (
        accuracy_ci(router_values)
    )

    base_est, base_low, base_high = (
        accuracy_ci(base_values)
    )

    strong_est, strong_low, strong_high = (
        accuracy_ci(strong_values)
    )


    # Paired JEV-router improvement
    # over the base model.
    (
        router_minus_base,
        router_minus_base_low,
        router_minus_base_high,
    ) = bootstrap_difference(
        router_values,
        base_values,
    )


    bootstrap_summary_rows.append(
        {
            "threshold": threshold,

            "router_accuracy": router_est,
            "router_accuracy_ci_low": router_low,
            "router_accuracy_ci_high": router_high,

            "base_accuracy": base_est,
            "base_accuracy_ci_low": base_low,
            "base_accuracy_ci_high": base_high,

            "strong_accuracy": strong_est,
            "strong_accuracy_ci_low": strong_low,
            "strong_accuracy_ci_high": strong_high,

            "router_minus_base": (
                router_minus_base
            ),

            "router_minus_base_ci_low": (
                router_minus_base_low
            ),

            "router_minus_base_ci_high": (
                router_minus_base_high
            ),
        }
    )


bootstrap_summary = pd.DataFrame(
    bootstrap_summary_rows
)

bootstrap_summary.to_csv(
    RESULTS_DIR
    / "e10_bootstrap_summary.csv",
    index=False,
)


# ============================================================
# ERROR-CAPTURE ANALYSIS
# ============================================================

error_capture_rows = []

for threshold in THRESHOLDS:

    subset = case_level_results[
        case_level_results["threshold"]
        == threshold
    ]

    escalated = (
        subset["jev_escalate"]
        == 1
    )

    base_wrong = (
        subset["stage1_correct"]
        == 0
    )

    strong_correct = (
        subset["strong_correct"]
        == 1
    )

    # Number of actual Qwen errors
    total_base_errors = int(
        base_wrong.sum()
    )

    # Actual Qwen errors that were escalated
    captured_errors = int(
        (
            escalated
            & base_wrong
        ).sum()
    )

    # Actual Qwen errors fixed by escalation
    corrected_errors = int(
        (
            escalated
            & base_wrong
            & strong_correct
        ).sum()
    )

    error_capture_rate = (
        captured_errors
        / total_base_errors
        if total_base_errors > 0
        else np.nan
    )

    error_correction_rate = (
        corrected_errors
        / total_base_errors
        if total_base_errors > 0
        else np.nan
    )

    error_capture_rows.append(
        {
            "threshold": threshold,
            "total_base_errors": total_base_errors,
            "captured_base_errors": captured_errors,
            "corrected_base_errors": corrected_errors,
            "error_capture_rate": error_capture_rate,
            "error_correction_rate": error_correction_rate,
        }
    )


error_capture = pd.DataFrame(
    error_capture_rows
)

error_capture.to_csv(
    RESULTS_DIR
    / "e10_error_capture.csv",
    index=False,
)


# ============================================================
# DOMAIN / DIFFICULTY ANALYSIS AT 0.90
# ============================================================

TARGET_THRESHOLD = 0.90

target = case_level_results[
    case_level_results["threshold"]
    == TARGET_THRESHOLD
].copy()


domain_rows = []

for domain, group in target.groupby(
    "domain"
):

    domain_rows.append(
        {
            "domain": domain,
            "n": len(group),

            "base_accuracy": (
                group["stage1_correct"]
                .mean()
            ),

            "strong_accuracy": (
                group["strong_correct"]
                .mean()
            ),

            "router_accuracy": (
                group["jev_router_correct"]
                .mean()
            ),

            "escalation_rate": (
                group["jev_escalate"]
                .mean()
            ),

            "base_errors": int(
                (
                    group["stage1_correct"]
                    == 0
                ).sum()
            ),

            "router_errors": int(
                (
                    group["jev_router_correct"]
                    == 0
                ).sum()
            ),
        }
    )


difficulty_rows = []

for difficulty, group in target.groupby(
    "difficulty"
):

    difficulty_rows.append(
        {
            "difficulty": difficulty,
            "n": len(group),

            "base_accuracy": (
                group["stage1_correct"]
                .mean()
            ),

            "strong_accuracy": (
                group["strong_correct"]
                .mean()
            ),

            "router_accuracy": (
                group["jev_router_correct"]
                .mean()
            ),

            "escalation_rate": (
                group["jev_escalate"]
                .mean()
            ),

            "base_errors": int(
                (
                    group["stage1_correct"]
                    == 0
                ).sum()
            ),

            "router_errors": int(
                (
                    group["jev_router_correct"]
                    == 0
                ).sum()
            ),
        }
    )


pd.DataFrame(
    domain_rows
).to_csv(
    RESULTS_DIR
    / "e10_domain_summary_090.csv",
    index=False,
)

pd.DataFrame(
    difficulty_rows
).to_csv(
    RESULTS_DIR
    / "e10_difficulty_summary_090.csv",
    index=False,
)


# ============================================================
# FINAL TERMINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("E10 COMPUTATION ALLOCATION ANALYSIS COMPLETE")
print("=" * 70)

print()
print("DATASET")
print("-" * 70)
print(f"Cases:                 {len(df)}")
print(
    f"Label balance:         "
    f"{int((df['label'] == 0).sum())} false / "
    f"{int((df['label'] == 1).sum())} true"
)

print()
print("BASELINES")
print("-" * 70)
print(
    f"Always-Base accuracy:  "
    f"{always_base_accuracy:.4f}"
)
print(
    f"Always-Strong accuracy:"
    f" {always_strong_accuracy:.4f}"
)

print()
print("JEV CONTROLLER")
print("-" * 70)
print(
    f"Mean P(Stage-1 correct): "
    f"{p.mean():.4f}"
)
print(
    f"Brier score:             "
    f"{brier:.6f}"
)
print(
    f"Log loss:                "
    f"{log_loss:.6f}"
)
print(
    f"ECE:                     "
    f"{ece:.6f}"
)

print()
print("ROUTER RESULTS")
print("-" * 70)

display_columns = [
    "threshold",
    "escalated_cases",
    "escalation_rate",
    "jev_router_accuracy",
    "random_router_expected_accuracy",
    "jev_router_minus_random",
    "total_jev_calls",
    "total_jev_cost_usd",
]

print(
    threshold_analysis[
        display_columns
    ].to_string(index=False)
)

print()
print("ERROR CAPTURE")
print("-" * 70)

print(
    error_capture.to_string(
        index=False
    )
)

print()
print("FILES SAVED")
print("-" * 70)

for filename in [
    "e10_controller_calibration.csv",
    "e10_policy_summary.csv",
    "e10_threshold_analysis.csv",
    "e10_case_level_results.csv",
    "e10_bootstrap_summary.csv",
    "e10_error_capture.csv",
    "e10_domain_summary_090.csv",
    "e10_difficulty_summary_090.csv",
]:

    print(
        RESULTS_DIR / filename
    )

print("=" * 70)
