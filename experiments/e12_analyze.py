import os
import numpy as np
import pandas as pd

DATA_PATH = "experiments/data/e12_temporal_stability_results.csv"
OUT_DIR = "experiments/results"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# 1. Reshape T1/T2/T3 into one row per independent case
# ------------------------------------------------------------

wide = (
    df.pivot(
        index=["case_id", "domain", "difficulty", "label"],
        columns="timepoint",
        values="probability",
    )
    .reset_index()
)

wide.columns.name = None

for t in ["T1", "T2", "T3"]:
    if t not in wide.columns:
        raise ValueError(f"Missing timepoint: {t}")

wide["d12"] = (wide["T2"] - wide["T1"]).abs()
wide["d23"] = (wide["T3"] - wide["T2"]).abs()
wide["d13"] = (wide["T3"] - wide["T1"]).abs()

wide["range"] = wide[["T1", "T2", "T3"]].max(axis=1) - \
                wide[["T1", "T2", "T3"]].min(axis=1)

wide["mean_pairwise_abs_change"] = (
    wide["d12"] + wide["d23"] + wide["d13"]
) / 3

wide["sd"] = wide[["T1", "T2", "T3"]].std(axis=1, ddof=1)

# Binary decisions at 0.5
for t in ["T1", "T2", "T3"]:
    wide[f"pred_{t}"] = (wide[t] >= 0.5).astype(int)

wide["flip_12"] = (wide["pred_T1"] != wide["pred_T2"]).astype(int)
wide["flip_23"] = (wide["pred_T2"] != wide["pred_T3"]).astype(int)
wide["flip_any"] = (
    (wide["pred_T1"] != wide["pred_T2"]) |
    (wide["pred_T2"] != wide["pred_T3"])
).astype(int)

# ------------------------------------------------------------
# 2. Basic temporal statistics
# ------------------------------------------------------------

print("=" * 70)
print("E12 TEMPORAL STABILITY ANALYSIS")
print("=" * 70)

print("\nCases:", len(wide))

print("\nAccuracy by timepoint:")
for t in ["T1", "T2", "T3"]:
    acc = (wide[f"pred_{t}"] == wide["label"]).mean()
    print(f"{t}: {acc:.4f}")

print("\nMean probability:")
for t in ["T1", "T2", "T3"]:
    print(f"{t}: {wide[t].mean():.4f}")

print("\nTemporal probability changes:")
print(f"T1 -> T2 mean |Δp|: {wide['d12'].mean():.6f}")
print(f"T2 -> T3 mean |Δp|: {wide['d23'].mean():.6f}")
print(f"T1 -> T3 mean |Δp|: {wide['d13'].mean():.6f}")

print("\nDistribution of case-level temporal range:")
print(f"Mean:   {wide['range'].mean():.6f}")
print(f"Median: {wide['range'].median():.6f}")
print(f"Max:    {wide['range'].max():.6f}")

print("\nDistribution of within-case SD:")
print(f"Mean:   {wide['sd'].mean():.6f}")
print(f"Median: {wide['sd'].median():.6f}")
print(f"Max:    {wide['sd'].max():.6f}")

print("\nBinary decision switching:")
print(f"T1 -> T2 flips: {wide['flip_12'].sum()}/{len(wide)} "
      f"({wide['flip_12'].mean():.4%})")

print(f"T2 -> T3 flips: {wide['flip_23'].sum()}/{len(wide)} "
      f"({wide['flip_23'].mean():.4%})")

print(f"Any temporal flip: {wide['flip_any'].sum()}/{len(wide)} "
      f"({wide['flip_any'].mean():.4%})")

# ------------------------------------------------------------
# 3. Domain and difficulty breakdown
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("BY DOMAIN")
print("-" * 70)

domain_summary = (
    wide.groupby("domain")
    .agg(
        n=("case_id", "count"),
        mean_abs_T1_T2=("d12", "mean"),
        mean_abs_T2_T3=("d23", "mean"),
        mean_range=("range", "mean"),
        median_range=("range", "median"),
        max_range=("range", "max"),
        any_flip=("flip_any", "mean"),
    )
    .reset_index()
)

print(domain_summary.to_string(index=False))

print("\n" + "-" * 70)
print("BY DIFFICULTY")
print("-" * 70)

difficulty_summary = (
    wide.groupby("difficulty")
    .agg(
        n=("case_id", "count"),
        mean_abs_T1_T2=("d12", "mean"),
        mean_abs_T2_T3=("d23", "mean"),
        mean_range=("range", "mean"),
        median_range=("range", "median"),
        max_range=("range", "max"),
        any_flip=("flip_any", "mean"),
    )
    .reset_index()
)

print(difficulty_summary.to_string(index=False))

# ------------------------------------------------------------
# 4. Bootstrap over independent CASES
# ------------------------------------------------------------

rng = np.random.default_rng(42)
B = 10000
n = len(wide)

def bootstrap_mean(column):
    values = wide[column].to_numpy()
    boot = np.empty(B)

    for i in range(B):
        sample = rng.integers(0, n, n)
        boot[i] = values[sample].mean()

    return (
        values.mean(),
        np.percentile(boot, 2.5),
        np.percentile(boot, 97.5),
    )

print("\n" + "-" * 70)
print("BOOTSTRAP 95% CIs — 10,000 RESAMPLES")
print("-" * 70)

for column, label in [
    ("d12", "T1 -> T2 mean |Δp|"),
    ("d23", "T2 -> T3 mean |Δp|"),
    ("d13", "T1 -> T3 mean |Δp|"),
    ("range", "Mean case probability range"),
    ("sd", "Mean within-case SD"),
    ("flip_any", "Any temporal decision flip"),
]:
    mean, lo, hi = bootstrap_mean(column)
    print(f"{label}: {mean:.6f} [{lo:.6f}, {hi:.6f}]")

# ------------------------------------------------------------
# 5. Hysteresis analysis
#
# Primary threshold = 0.90, matching the earlier controller work.
# Hysteresis band:
#   enter/accept >= 0.95
#   leave/accept <= 0.85
#
# State meaning:
#   1 = trust / accept
#   0 = escalate / do not trust
# ------------------------------------------------------------

def no_hysteresis_states(values, threshold):
    return (values >= threshold).astype(int)


def hysteresis_states(values, low, high):
    states = []

    # Initialize from first observation.
    if values[0] >= high:
        state = 1
    elif values[0] <= low:
        state = 0
    else:
        # Conservative initialization inside the band.
        state = 0

    states.append(state)

    for value in values[1:]:
        if value >= high:
            state = 1
        elif value <= low:
            state = 0
        # Otherwise retain previous state.

        states.append(state)

    return np.array(states)


def evaluate_controller(states):
    states = np.asarray(states)

    switches = np.sum(states[1:] != states[:-1])

    # Ground-truth desired state:
    # true proposition -> accept/trust
    # false proposition -> escalate
    desired = None

    return switches


# Evaluate each case independently.
thresholds = [0.50, 0.70, 0.90, 0.95]

hysteresis_results = []

for threshold in thresholds:

    # No hysteresis
    no_hyst_switches = []
    no_hyst_errors = []

    # Symmetric +/- 0.05 hysteresis
    low = max(0.0, threshold - 0.05)
    high = min(1.0, threshold + 0.05)

    hyst_switches = []
    hyst_errors = []

    for _, row in wide.iterrows():

        values = np.array([row["T1"], row["T2"], row["T3"]])
        label = int(row["label"])

        desired = np.full(3, label)

        # No hysteresis
        states_no = no_hysteresis_states(values, threshold)

        no_hyst_switches.append(
            np.sum(states_no[1:] != states_no[:-1])
        )

        no_hyst_errors.append(
            np.mean(states_no != desired)
        )

        # Hysteresis
        states_h = hysteresis_states(values, low, high)

        hyst_switches.append(
            np.sum(states_h[1:] != states_h[:-1])
        )

        hyst_errors.append(
            np.mean(states_h != desired)
        )

    hysteresis_results.append({
        "threshold": threshold,
        "hysteresis_low": low,
        "hysteresis_high": high,
        "no_hysteresis_total_switches": sum(no_hyst_switches),
        "hysteresis_total_switches": sum(hyst_switches),
        "no_hysteresis_mean_switches": np.mean(no_hyst_switches),
        "hysteresis_mean_switches": np.mean(hyst_switches),
        "no_hysteresis_control_error": np.mean(no_hyst_errors),
        "hysteresis_control_error": np.mean(hyst_errors),
    })

hyst_df = pd.DataFrame(hysteresis_results)

print("\n" + "-" * 70)
print("HYSTERESIS ANALYSIS")
print("-" * 70)

print(hyst_df.to_string(index=False))

# ------------------------------------------------------------
# 6. Detailed primary hysteresis result
# ------------------------------------------------------------

primary = hyst_df[hyst_df["threshold"] == 0.90].iloc[0]

print("\nPrimary controller threshold: 0.90")
print("Hysteresis band: [0.85, 0.95]")

print(
    f"No hysteresis switches: "
    f"{int(primary['no_hysteresis_total_switches'])}"
)

print(
    f"Hysteresis switches: "
    f"{int(primary['hysteresis_total_switches'])}"
)

print(
    f"No hysteresis control error: "
    f"{primary['no_hysteresis_control_error']:.4%}"
)

print(
    f"Hysteresis control error: "
    f"{primary['hysteresis_control_error']:.4%}"
)

# ------------------------------------------------------------
# 7. Save machine-readable outputs
# ------------------------------------------------------------

wide.to_csv(
    os.path.join(OUT_DIR, "e12_temporal_case_level.csv"),
    index=False,
)

domain_summary.to_csv(
    os.path.join(OUT_DIR, "e12_domain_summary.csv"),
    index=False,
)

difficulty_summary.to_csv(
    os.path.join(OUT_DIR, "e12_difficulty_summary.csv"),
    index=False,
)

hyst_df.to_csv(
    os.path.join(OUT_DIR, "e12_hysteresis_summary.csv"),
    index=False,
)

print("\nSaved:")
print("experiments/results/e12_temporal_case_level.csv")
print("experiments/results/e12_domain_summary.csv")
print("experiments/results/e12_difficulty_summary.csv")
print("experiments/results/e12_hysteresis_summary.csv")

print("\n" + "=" * 70)
print("E12 ANALYSIS COMPLETE")
print("=" * 70)
