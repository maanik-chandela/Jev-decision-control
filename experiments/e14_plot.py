from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# E14 — State Attack Robustness Plots
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "experiments" / "results"

ATTACK_SUMMARY = RESULTS_DIR / "e14_attack_summary.csv"
PAIRED_RESULTS = RESULTS_DIR / "e14_paired_case_results.csv"

# Output figures
FIG1 = RESULTS_DIR / "e14_attack_probability_shift.png"
FIG2 = RESULTS_DIR / "e14_case_level_attack_shifts.png"
FIG3 = RESULTS_DIR / "e14_confidence_vs_attack_shift.png"


# ============================================================
# Load results
# ============================================================

attack_summary = pd.read_csv(ATTACK_SUMMARY)
paired = pd.read_csv(PAIRED_RESULTS)

# Consistent ordering
attack_order = [
    "prompt_injection",
    "instruction_override",
    "authority_impersonation",
    "irrelevant_malicious",
    "adversarial_evidence",
]

paired["attack_condition"] = pd.Categorical(
    paired["attack_condition"],
    categories=attack_order,
    ordered=True,
)

attack_summary["attack_condition"] = pd.Categorical(
    attack_summary["attack_condition"],
    categories=attack_order,
    ordered=True,
)

paired = paired.sort_values(
    ["attack_condition", "case_id"]
)

attack_summary = attack_summary.sort_values(
    "attack_condition"
)


# ============================================================
# Friendly labels
# ============================================================

labels = {
    "prompt_injection": "Prompt injection",
    "instruction_override": "Instruction override",
    "authority_impersonation": "Authority impersonation",
    "irrelevant_malicious": "Irrelevant malicious",
    "adversarial_evidence": "Adversarial evidence",
}


# ============================================================
# Figure 1
# Mean absolute probability shift by attack type
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(attack_summary))

means = attack_summary["mean_abs_delta_p"].to_numpy()
ci_low = attack_summary["abs_delta_ci_low"].to_numpy()
ci_high = attack_summary["abs_delta_ci_high"].to_numpy()

yerr = np.vstack([
    means - ci_low,
    ci_high - means,
])

bars = ax.bar(
    x,
    means,
    yerr=yerr,
    capsize=5,
)

ax.set_xticks(x)
ax.set_xticklabels(
    [
        labels[c]
        for c in attack_summary["attack_condition"]
    ],
    rotation=20,
    ha="right",
)

ax.set_ylabel("Mean absolute probability shift |Δp|")
ax.set_title(
    "E14: Probability sensitivity to state attacks"
)

ax.set_ylim(
    0,
    max(ci_high) * 1.20
)

ax.grid(
    axis="y",
    alpha=0.25,
)

# Annotate bars
for i, value in enumerate(means):
    ax.text(
        i,
        value + 0.008,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

fig.tight_layout()

fig.savefig(
    FIG1,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 2
# Case-level attack probability shifts
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

# Use jitter so the 60 cases don't completely overlap
rng = np.random.default_rng(42)

for i, attack in enumerate(attack_order):

    group = paired[
        paired["attack_condition"] == attack
    ].copy()

    y = group["abs_probability_delta"].to_numpy()

    jitter = rng.uniform(
        -0.12,
        0.12,
        size=len(y),
    )

    x_values = np.full(
        len(y),
        i,
        dtype=float,
    ) + jitter

    ax.scatter(
        x_values,
        y,
        alpha=0.65,
        s=32,
    )

    # Mean marker
    mean_value = y.mean()

    ax.scatter(
        i,
        mean_value,
        marker="D",
        s=70,
        zorder=5,
    )


# Highlight E14_D04
d04 = paired[
    paired["case_id"] == "E14_D04"
]

for _, row in d04.iterrows():

    attack = row["attack_condition"]

    if attack not in attack_order:
        continue

    x_position = attack_order.index(attack)

    ax.annotate(
        "E14_D04",
        (
            x_position,
            row["abs_probability_delta"],
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )


ax.set_xticks(
    range(len(attack_order))
)

ax.set_xticklabels(
    [
        labels[a]
        for a in attack_order
    ],
    rotation=20,
    ha="right",
)

ax.set_ylabel(
    "Absolute probability shift |Δp|"
)

ax.set_title(
    "E14: Case-level probability shifts relative to clean state"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIG2,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 3
# Clean confidence vs attack-induced probability shift
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

for attack in attack_order:

    group = paired[
        paired["attack_condition"] == attack
    ].copy()

    x_values = group["clean_confidence"].to_numpy()
    y_values = group["abs_probability_delta"].to_numpy()

    ax.scatter(
        x_values,
        y_values,
        alpha=0.55,
        s=30,
        label=labels[attack],
    )


# Mark E14_D04 across all attack conditions
d04 = paired[
    paired["case_id"] == "E14_D04"
]

ax.scatter(
    d04["clean_confidence"],
    d04["abs_probability_delta"],
    marker="*",
    s=180,
    zorder=10,
    label="E14_D04",
)

ax.axvline(
    0.75,
    linestyle="--",
    alpha=0.35,
)

ax.axvline(
    0.90,
    linestyle="--",
    alpha=0.35,
)

ax.set_xlabel(
    "Clean-state confidence"
)

ax.set_ylabel(
    "Absolute probability shift |Δp|"
)

ax.set_title(
    "E14: Attack sensitivity as a function of clean confidence"
)

ax.set_xlim(
    0.45,
    1.01,
)

ax.set_ylim(
    0,
    max(paired["abs_probability_delta"]) * 1.12,
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    fontsize=8,
    loc="upper right",
)

fig.tight_layout()

fig.savefig(
    FIG3,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Final output
# ============================================================

print("=" * 70)
print("E14 PLOTTING COMPLETE")
print("=" * 70)

print("\nCreated figures:")

print(
    f"  {FIG1.relative_to(ROOT)}"
)

print(
    f"  {FIG2.relative_to(ROOT)}"
)

print(
    f"  {FIG3.relative_to(ROOT)}"
)

print("\nAll figures saved at 300 DPI.")