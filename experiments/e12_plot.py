import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "experiments", "results")

case_df = pd.read_csv(
    os.path.join(DATA_DIR, "e12_temporal_case_level.csv")
)

domain_df = pd.read_csv(
    os.path.join(DATA_DIR, "e12_domain_summary.csv")
)

difficulty_df = pd.read_csv(
    os.path.join(DATA_DIR, "e12_difficulty_summary.csv")
)

hyst_df = pd.read_csv(
    os.path.join(DATA_DIR, "e12_hysteresis_summary.csv")
)


# ============================================================
# FIGURE 1
# Case-level temporal probability range
# ============================================================

plot_df = case_df.sort_values(
    "range",
    ascending=True
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5.5))

x = np.arange(1, len(plot_df) + 1)

ax.plot(
    x,
    plot_df["range"],
    marker="o",
    markersize=3,
    linewidth=1.5
)

ax.axhline(
    plot_df["range"].median(),
    linestyle="--",
    linewidth=1.5,
    label=f"Median = {plot_df['range'].median():.3f}"
)

ax.axhline(
    plot_df["range"].mean(),
    linestyle=":",
    linewidth=1.5,
    label=f"Mean = {plot_df['range'].mean():.3f}"
)

ax.set_title("E12 — Case-Level Temporal Probability Variation")
ax.set_xlabel("Cases ordered by temporal probability range")
ax.set_ylabel("Probability range (max − min)")
ax.set_xlim(1, len(plot_df))
ax.set_ylim(0, max(plot_df["range"]) * 1.15)
ax.grid(axis="y", alpha=0.25)
ax.legend()

plt.tight_layout()

fig1 = os.path.join(
    DATA_DIR,
    "e12_case_level_temporal_range.png"
)

plt.savefig(
    fig1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# Temporal probability range by difficulty
# ============================================================

difficulty_order = [
    "easy",
    "moderate",
    "hard"
]

plot_difficulty = (
    difficulty_df
    .set_index("difficulty")
    .loc[difficulty_order]
)

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.bar(
    ["Easy", "Moderate", "Hard"],
    plot_difficulty["mean_range"]
)

ax.set_title(
    "E12 — Temporal Probability Variation by Difficulty"
)

ax.set_xlabel("Difficulty")
ax.set_ylabel("Mean case-level probability range")

ax.set_ylim(
    0,
    max(plot_difficulty["mean_range"]) * 1.25
)

ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

fig2 = os.path.join(
    DATA_DIR,
    "e12_probability_range_by_difficulty.png"
)

plt.savefig(
    fig2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# Temporal probability range by domain
# ============================================================

domain_order = [
    "mathematics",
    "probability",
    "formal_logic",
    "science_reasoning",
    "data_reasoning"
]

plot_domain = (
    domain_df
    .set_index("domain")
    .loc[domain_order]
)

domain_labels = [
    "Mathematics",
    "Probability",
    "Formal logic",
    "Science reasoning",
    "Data reasoning"
]

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.bar(
    domain_labels,
    plot_domain["mean_range"]
)

ax.set_title(
    "E12 — Temporal Probability Variation by Domain"
)

ax.set_xlabel("Domain")
ax.set_ylabel("Mean case-level probability range")

ax.set_ylim(
    0,
    max(plot_domain["mean_range"]) * 1.25
)

ax.tick_params(
    axis="x",
    rotation=20
)

ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

fig3 = os.path.join(
    DATA_DIR,
    "e12_probability_range_by_domain.png"
)

plt.savefig(
    fig3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4
# Hysteresis switching
# ============================================================

threshold_labels = [
    f"{x:.2f}"
    for x in hyst_df["threshold"]
]

x = np.arange(len(threshold_labels))
width = 0.36

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.bar(
    x - width / 2,
    hyst_df["no_hysteresis_mean_switches"],
    width,
    label="No hysteresis"
)

ax.bar(
    x + width / 2,
    hyst_df["hysteresis_mean_switches"],
    width,
    label="Hysteresis"
)

ax.set_title(
    "E12 — Hysteresis Reduces Control-State Switching"
)

ax.set_xlabel("Decision threshold")
ax.set_ylabel("Mean switches per case")

ax.set_xticks(x)
ax.set_xticklabels(threshold_labels)

ax.grid(axis="y", alpha=0.25)
ax.legend()

plt.tight_layout()

fig4 = os.path.join(
    DATA_DIR,
    "e12_hysteresis_switching.png"
)

plt.savefig(
    fig4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 5
# Hysteresis control error
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(
    threshold_labels,
    hyst_df["no_hysteresis_control_error"] * 100,
    marker="o",
    linewidth=2,
    label="No hysteresis"
)

ax.plot(
    threshold_labels,
    hyst_df["hysteresis_control_error"] * 100,
    marker="o",
    linewidth=2,
    label="Hysteresis"
)

ax.set_title(
    "E12 — Hysteresis Control Error"
)

ax.set_xlabel("Decision threshold")
ax.set_ylabel("Control error (%)")

ax.grid(axis="y", alpha=0.25)
ax.legend()

plt.tight_layout()

fig5 = os.path.join(
    DATA_DIR,
    "e12_hysteresis_control_error.png"
)

plt.savefig(
    fig5,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("E12 FIGURES CREATED")
print("=" * 70)

print(fig1)
print(fig2)
print(fig3)
print(fig4)
print(fig5)