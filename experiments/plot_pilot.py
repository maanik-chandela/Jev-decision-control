import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "experiments" / "data" / "pilot_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# -----------------------------
# Derived variables
# -----------------------------

df["correct"] = (
    df["predicted_label"] == df["true_label"]
).astype(int)

df["confidence"] = df["jev_probability"].apply(
    lambda p: p if p >= 0.5 else 1 - p
)

# -----------------------------
# 1. Probability distribution
# -----------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df.loc[df["true_label"] == 1, "jev_probability"],
    bins=10,
    alpha=0.7,
    label="True cases"
)

plt.hist(
    df.loc[df["true_label"] == 0, "jev_probability"],
    bins=10,
    alpha=0.7,
    label="False cases"
)

plt.xlabel("JEV probability of True")
plt.ylabel("Number of cases")
plt.title("Pilot Probability Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pilot_probability_distribution.png",
    dpi=300
)

plt.close()

# -----------------------------
# 2. Reliability diagram
# -----------------------------

bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

df["prob_bin"] = pd.cut(
    df["jev_probability"],
    bins=bins,
    include_lowest=True,
    labels=False
)

reliability = (
    df.groupby("prob_bin", observed=True)
    .agg(
        mean_probability=("jev_probability", "mean"),
        empirical_accuracy=("true_label", "mean"),
        count=("true_label", "size")
    )
    .dropna()
)

plt.figure(figsize=(6, 6))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.scatter(
    reliability["mean_probability"],
    reliability["empirical_accuracy"],
    s=reliability["count"] * 12,
    label="JEV pilot"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.xlabel("Mean predicted probability")
plt.ylabel("Empirical frequency of True")
plt.title("Pilot Reliability Diagram")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pilot_reliability_diagram.png",
    dpi=300
)

plt.close()

# -----------------------------
# 3. Confidence distribution
# -----------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["confidence"],
    bins=10
)

plt.xlabel("Prediction confidence")
plt.ylabel("Number of cases")
plt.title("Pilot Confidence Distribution")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pilot_confidence_distribution.png",
    dpi=300
)

plt.close()

# -----------------------------
# 4. Probability by difficulty
# -----------------------------

difficulty_order = [
    difficulty
    for difficulty in ["easy", "medium", "hard"]
    if difficulty in df["difficulty"].unique()
]

groups = [
    df.loc[
        df["difficulty"] == difficulty,
        "jev_probability"
    ]
    for difficulty in difficulty_order
]

plt.figure(figsize=(7, 5))

plt.boxplot(
    groups,
    tick_labels=difficulty_order
)

plt.xlabel("Difficulty")
plt.ylabel("JEV probability of True")
plt.title("JEV Probability by Difficulty")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pilot_probability_by_difficulty.png",
    dpi=300
)

plt.close()

# -----------------------------
# 5. Probability by domain
# -----------------------------

domain_order = sorted(df["domain"].unique())

groups = [
    df.loc[
        df["domain"] == domain,
        "jev_probability"
    ]
    for domain in domain_order
]

plt.figure(figsize=(9, 5))

plt.boxplot(
    groups,
    tick_labels=domain_order
)

plt.xlabel("Domain")
plt.ylabel("JEV probability of True")
plt.title("JEV Probability by Domain")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pilot_probability_by_domain.png",
    dpi=300
)

plt.close()

# -----------------------------
# Save reliability table
# -----------------------------

reliability.to_csv(
    OUTPUT_DIR / "pilot_reliability_table.csv"
)

print()
print("Pilot figures created successfully.")
print()

for file in sorted(OUTPUT_DIR.glob("pilot_*.png")):
    print(file)

print()
print("Reliability table:")
print(
    OUTPUT_DIR / "pilot_reliability_table.csv"
)