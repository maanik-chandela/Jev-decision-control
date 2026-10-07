import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE_DIR = "/Users/maanikchandela/JEV-Research/jev-decision-control"

DATA_FILE = os.path.join(
    BASE_DIR,
    "experiments/data/e13_independent_results.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "experiments/results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_FILE)

df["representation"] = (
    df["representation"]
    .str.capitalize()
)

print("Loaded:", len(df), "rows")


# ============================================================
# Pivot to case level
# ============================================================

prob = df.pivot_table(
    index="case_id",
    columns="representation",
    values="probability_true",
    aggfunc="first"
)

prob = prob[["Noul", "Choice", "Score"]]

prob["spread"] = (
    prob.max(axis=1)
    -
    prob.min(axis=1)
)

# Preserve original case ordering
case_order = (
    df[["case_id"]]
    .drop_duplicates()["case_id"]
    .tolist()
)

prob = prob.reindex(case_order)


# ============================================================
# Figure 1
# Probability distribution by decision type
# ============================================================

plt.figure(figsize=(8, 6))

data = [
    prob["Noul"].values,
    prob["Choice"].values,
    prob["Score"].values
]

plt.boxplot(
    data,
    tick_labels=["Noul", "Choice", "Score"]
)

plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1
)

plt.ylabel("Probability of True")
plt.title(
    "E13: Probability Outputs by Decision Type"
)

plt.tight_layout()

path1 = os.path.join(
    RESULTS_DIR,
    "e13_probability_by_decision_type.png"
)

plt.savefig(
    path1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path1)


# ============================================================
# Figure 2
# Case-level representation spread
# ============================================================

ranked = prob.sort_values(
    "spread",
    ascending=False
).copy()

plt.figure(figsize=(11, 6))

x = np.arange(len(ranked))

plt.bar(
    x,
    ranked["spread"]
)

plt.axhline(
    ranked["spread"].mean(),
    linestyle="--",
    linewidth=1,
    label=f"Mean = {ranked['spread'].mean():.3f}"
)

plt.xticks(
    x,
    ranked.index,
    rotation=90,
    fontsize=7
)

plt.ylabel(
    "Probability spread (max − min)"
)

plt.xlabel("Case")

plt.title(
    "E13: Case-Level Sensitivity to Decision Type"
)

plt.legend()

plt.tight_layout()

path2 = os.path.join(
    RESULTS_DIR,
    "e13_case_level_representation_spread.png"
)

plt.savefig(
    path2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path2)


# ============================================================
# Figure 3
# Representation spread by domain
# ============================================================

domain_map = (
    df.groupby("case_id")["domain"]
    .first()
)

prob["domain"] = domain_map

domain_summary = (
    prob.groupby("domain")["spread"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))

plt.bar(
    domain_summary.index,
    domain_summary.values
)

plt.ylabel(
    "Mean probability spread"
)

plt.xlabel("Domain")

plt.title(
    "E13: Decision-Type Sensitivity Across Domains"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

path3 = os.path.join(
    RESULTS_DIR,
    "e13_representation_spread_by_domain.png"
)

plt.savefig(
    path3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path3)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("E13 PLOTS COMPLETE")
print("=" * 70)

print(f"Mean representation spread: {prob['spread'].mean():.4f}")
print(f"Median representation spread: {prob['spread'].median():.4f}")
print(f"Maximum representation spread: {prob['spread'].max():.4f}")

print("\nTop 5 cases:")
print(
    prob[
        ["Noul", "Choice", "Score", "spread"]
    ]
    .sort_values("spread", ascending=False)
    .head(5)
    .to_string()
)