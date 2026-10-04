import os
import numpy as np
import pandas as pd

INPUT_FILE = "experiments/data/e9_state_manipulation_results.csv"
RESULTS_DIR = "experiments/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df = df[df["probability"].notna()].copy()

# ------------------------------------------------------------
# Basic summary
# ------------------------------------------------------------

print("=" * 70)
print("E9 STATE MANIPULATION ROBUSTNESS")
print("=" * 70)

print(f"Evaluations: {len(df)}")
print(f"Cases: {df['case_id'].nunique()}")
print(f"Domains: {df['domain'].nunique()}")
print()

print("Accuracy by condition:")
print(df.groupby("condition")["correct"].mean().round(4))
print()

print("Mean probability by condition:")
print(df.groupby("condition")["probability"].mean().round(4))
print()

print("Mean confidence by condition:")
print(df.groupby("condition")["confidence"].mean().round(4))
print()

# ------------------------------------------------------------
# Pivot to matched 60-case structure
# ------------------------------------------------------------

prob = df.pivot(
    index="case_id",
    columns="condition",
    values="probability"
)

conf = df.pivot(
    index="case_id",
    columns="condition",
    values="confidence"
)

correct = df.pivot(
    index="case_id",
    columns="condition",
    values="correct"
)

labels = df.drop_duplicates("case_id").set_index("case_id")["label"]
domains = df.drop_duplicates("case_id").set_index("case_id")["domain"]

matched = pd.DataFrame({
    "label": labels,
    "domain": domains,
    "original_p": prob["original"],
    "irrelevant_p": prob["irrelevant_context"],
    "distracting_p": prob["distracting_context"],
    "original_conf": conf["original"],
    "irrelevant_conf": conf["irrelevant_context"],
    "distracting_conf": conf["distracting_context"],
    "original_correct": correct["original"],
    "irrelevant_correct": correct["irrelevant_context"],
    "distracting_correct": correct["distracting_context"],
})

# ------------------------------------------------------------
# Pairwise probability changes
# ------------------------------------------------------------

matched["original_to_irrelevant"] = (
    matched["irrelevant_p"] - matched["original_p"]
)

matched["original_to_distracting"] = (
    matched["distracting_p"] - matched["original_p"]
)

matched["irrelevant_to_distracting"] = (
    matched["distracting_p"] - matched["irrelevant_p"]
)

matched["abs_original_to_irrelevant"] = (
    matched["original_to_irrelevant"].abs()
)

matched["abs_original_to_distracting"] = (
    matched["original_to_distracting"].abs()
)

matched["abs_irrelevant_to_distracting"] = (
    matched["irrelevant_to_distracting"].abs()
)

matched["probability_spread"] = (
    matched[["original_p", "irrelevant_p", "distracting_p"]].max(axis=1)
    - matched[["original_p", "irrelevant_p", "distracting_p"]].min(axis=1)
)

# ------------------------------------------------------------
# Decision changes
# ------------------------------------------------------------

matched["original_pred"] = (matched["original_p"] >= 0.5).astype(int)
matched["irrelevant_pred"] = (matched["irrelevant_p"] >= 0.5).astype(int)
matched["distracting_pred"] = (matched["distracting_p"] >= 0.5).astype(int)

matched["decision_change_original_irrelevant"] = (
    matched["original_pred"] != matched["irrelevant_pred"]
)

matched["decision_change_original_distracting"] = (
    matched["original_pred"] != matched["distracting_pred"]
)

matched["any_decision_change"] = (
    (matched["original_pred"] != matched["irrelevant_pred"])
    | (matched["original_pred"] != matched["distracting_pred"])
)

# ------------------------------------------------------------
# Print pairwise summary
# ------------------------------------------------------------

pairs = {
    "original → irrelevant": "abs_original_to_irrelevant",
    "original → distracting": "abs_original_to_distracting",
    "irrelevant → distracting": "abs_irrelevant_to_distracting",
}

print("=" * 70)
print("PAIRWISE PROBABILITY SHIFTS")
print("=" * 70)

pair_rows = []

for name, col in pairs.items():
    signed_col = col.replace("abs_", "")

    values = matched[col]

    row = {
        "comparison": name,
        "mean_signed_change": matched[signed_col].mean(),
        "mean_abs_change": values.mean(),
        "median_abs_change": values.median(),
        "max_abs_change": values.max(),
    }

    if name == "original → irrelevant":
        row["decision_change_rate"] = (
            matched["decision_change_original_irrelevant"].mean()
        )
    elif name == "original → distracting":
        row["decision_change_rate"] = (
            matched["decision_change_original_distracting"].mean()
        )
    else:
        row["decision_change_rate"] = np.nan

    pair_rows.append(row)

pair_summary = pd.DataFrame(pair_rows)

print(pair_summary.round(4).to_string(index=False))

# ------------------------------------------------------------
# Case-level summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("CASE-LEVEL SUMMARY")
print("=" * 70)

print(
    f"Mean probability spread: "
    f"{matched['probability_spread'].mean():.4f}"
)

print(
    f"Median probability spread: "
    f"{matched['probability_spread'].median():.4f}"
)

print(
    f"Maximum probability spread: "
    f"{matched['probability_spread'].max():.4f}"
)

print(
    f"Cases with any binary decision change: "
    f"{matched['any_decision_change'].sum()}/{len(matched)} "
    f"({matched['any_decision_change'].mean():.4f})"
)

# ------------------------------------------------------------
# Domain effects
# ------------------------------------------------------------

domain_summary = (
    matched.groupby("domain")
    .agg(
        cases=("label", "size"),
        mean_spread=("probability_spread", "mean"),
        median_spread=("probability_spread", "median"),
        max_spread=("probability_spread", "max"),
        decision_changes=("any_decision_change", "sum"),
    )
    .reset_index()
)

print()
print("=" * 70)
print("DOMAIN SUMMARY")
print("=" * 70)
print(domain_summary.round(4).to_string(index=False))

# ------------------------------------------------------------
# Largest probability shifts
# ------------------------------------------------------------

largest = matched.sort_values(
    "probability_spread",
    ascending=False
).head(10)

print()
print("=" * 70)
print("10 LARGEST CASE-LEVEL PROBABILITY SPREADS")
print("=" * 70)

print(
    largest[
        [
            "domain",
            "label",
            "original_p",
            "irrelevant_p",
            "distracting_p",
            "probability_spread",
            "any_decision_change",
        ]
    ].round(4).to_string()
)

# ------------------------------------------------------------
# Bootstrap over 60 independent underlying cases
# ------------------------------------------------------------

rng = np.random.default_rng(42)
n_boot = 10000
n_cases = len(matched)

boot_rows = []

for _ in range(n_boot):
    idx = rng.integers(0, n_cases, size=n_cases)
    sample = matched.iloc[idx]

    boot_rows.append({
        "mean_spread": sample["probability_spread"].mean(),
        "median_spread": sample["probability_spread"].median(),
        "abs_original_irrelevant": sample[
            "abs_original_to_irrelevant"
        ].mean(),
        "abs_original_distracting": sample[
            "abs_original_to_distracting"
        ].mean(),
        "decision_change_original_irrelevant": sample[
            "decision_change_original_irrelevant"
        ].mean(),
        "decision_change_original_distracting": sample[
            "decision_change_original_distracting"
        ].mean(),
        "any_decision_change": sample[
            "any_decision_change"
        ].mean(),
    })

bootstrap = pd.DataFrame(boot_rows)

def ci(series):
    return (
        float(series.quantile(0.025)),
        float(series.quantile(0.975)),
    )

bootstrap_summary = pd.DataFrame([
    {
        "metric": "mean_probability_spread",
        "estimate": matched["probability_spread"].mean(),
        "ci_lower": ci(bootstrap["mean_spread"])[0],
        "ci_upper": ci(bootstrap["mean_spread"])[1],
    },
    {
        "metric": "median_probability_spread",
        "estimate": matched["probability_spread"].median(),
        "ci_lower": ci(bootstrap["median_spread"])[0],
        "ci_upper": ci(bootstrap["median_spread"])[1],
    },
    {
        "metric": "mean_abs_original_irrelevant",
        "estimate": matched["abs_original_to_irrelevant"].mean(),
        "ci_lower": ci(bootstrap["abs_original_irrelevant"])[0],
        "ci_upper": ci(bootstrap["abs_original_irrelevant"])[1],
    },
    {
        "metric": "mean_abs_original_distracting",
        "estimate": matched["abs_original_to_distracting"].mean(),
        "ci_lower": ci(bootstrap["abs_original_distracting"])[0],
        "ci_upper": ci(bootstrap["abs_original_distracting"])[1],
    },
    {
        "metric": "decision_change_original_irrelevant",
        "estimate": matched["decision_change_original_irrelevant"].mean(),
        "ci_lower": ci(
            bootstrap["decision_change_original_irrelevant"]
        )[0],
        "ci_upper": ci(
            bootstrap["decision_change_original_irrelevant"]
        )[1],
    },
    {
        "metric": "decision_change_original_distracting",
        "estimate": matched["decision_change_original_distracting"].mean(),
        "ci_lower": ci(
            bootstrap["decision_change_original_distracting"]
        )[0],
        "ci_upper": ci(
            bootstrap["decision_change_original_distracting"]
        )[1],
    },
    {
        "metric": "any_decision_change",
        "estimate": matched["any_decision_change"].mean(),
        "ci_lower": ci(bootstrap["any_decision_change"])[0],
        "ci_upper": ci(bootstrap["any_decision_change"])[1],
    },
])

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

matched.to_csv(
    f"{RESULTS_DIR}/e9_matched_cases.csv",
    index=True,
)

pair_summary.to_csv(
    f"{RESULTS_DIR}/e9_pairwise_probability_shifts.csv",
    index=False,
)

domain_summary.to_csv(
    f"{RESULTS_DIR}/e9_domain_summary.csv",
    index=False,
)

largest.to_csv(
    f"{RESULTS_DIR}/e9_largest_probability_spreads.csv",
    index=True,
)

bootstrap_summary.to_csv(
    f"{RESULTS_DIR}/e9_bootstrap_summary.csv",
    index=False,
)

bootstrap.to_csv(
    f"{RESULTS_DIR}/e9_bootstrap_samples.csv",
    index=False,
)

print()
print("=" * 70)
print("BOOTSTRAP SUMMARY")
print("=" * 70)
print(bootstrap_summary.round(4).to_string(index=False))

print()
print("Saved E9 analysis files to experiments/results/")
