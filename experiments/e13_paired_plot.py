import os
import pandas as pd
import matplotlib.pyplot as plt

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

# Load E13 independent results
df = pd.read_csv(DATA_FILE)

df["representation"] = (
    df["representation"]
    .str.capitalize()
)

# One row per underlying case, one column per representation
prob = df.pivot_table(
    index="case_id",
    columns="representation",
    values="probability_true",
    aggfunc="first"
)[["Noul", "Choice", "Score"]]

# Preserve original case ordering
case_order = (
    df[["case_id"]]
    .drop_duplicates()["case_id"]
    .tolist()
)

prob = prob.reindex(case_order)

# Calculate representation spread
prob["spread"] = (
    prob.max(axis=1)
    -
    prob.min(axis=1)
)

# Sort strongest sensitivity to weakest
prob = prob.sort_values(
    "spread",
    ascending=False
)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(12, 8))

x = [0, 1, 2]

# Every line represents ONE underlying proposition
for case_id, row in prob.iterrows():

    plt.plot(
        x,
        [
            row["Noul"],
            row["Choice"],
            row["Score"]
        ],
        marker="o",
        linewidth=1,
        alpha=0.30
    )

# Highlight the two strongest cases
for case_id in ["E13_D08", "E13_D10"]:

    if case_id in prob.index:

        row = prob.loc[case_id]

        plt.plot(
            x,
            [
                row["Noul"],
                row["Choice"],
                row["Score"]
            ],
            marker="o",
            linewidth=2.5,
            label=case_id
        )

# Binary decision boundary
plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="Decision boundary = 0.50"
)

plt.xticks(
    x,
    ["Noul", "Choice", "Score"]
)

plt.xlabel("Decision representation")
plt.ylabel("Probability of True")

plt.ylim(-0.02, 1.02)

plt.title(
    "E13: Paired Probability Outputs Across Decision Types"
)

plt.legend(loc="best")

plt.tight_layout()

output = os.path.join(
    RESULTS_DIR,
    "e13_paired_probability_outputs.png"
)

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 70)
print("E13 PAIRED PROBABILITY FIGURE")
print("=" * 70)

print("Cases plotted:", len(prob))
print(
    "Mean representation spread:",
    f"{prob['spread'].mean():.4f}"
)
print(
    "Median representation spread:",
    f"{prob['spread'].median():.4f}"
)
print(
    "Maximum representation spread:",
    f"{prob['spread'].max():.4f}"
)

print("\nSaved:")
print(output)