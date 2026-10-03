import pandas as pd
from pathlib import Path

INPUT = Path("experiments/data/e4c_h_baseline_results.csv")
OUTPUT = Path("experiments/data/e4c_h_selected_cases.csv")

df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# Frozen confidence thresholds
# ------------------------------------------------------------

MODERATE_LOW_MIN = 0.20
MODERATE_LOW_MAX = 0.80

# High confidence:
# probability <= 0.10 OR probability >= 0.90

# ------------------------------------------------------------
# Moderate/low cases
# ------------------------------------------------------------

moderate = df[
    (df["probability"] >= MODERATE_LOW_MIN)
    & (df["probability"] <= MODERATE_LOW_MAX)
].copy()

moderate = moderate.sort_values("id")

# ------------------------------------------------------------
# High-confidence candidates
# ------------------------------------------------------------

high = df[
    (df["probability"] <= 0.10)
    | (df["probability"] >= 0.90)
].copy()

high = high.sort_values("id")

# ------------------------------------------------------------
# One deterministic high-confidence control per domain
# ------------------------------------------------------------

selected_high = []

for domain in sorted(high["domain"].unique()):
    candidates = high[high["domain"] == domain].sort_values("id")

    if len(candidates) == 0:
        raise RuntimeError(f"No high-confidence candidate for {domain}")

    selected_high.append(candidates.iloc[0])

selected_high = pd.DataFrame(selected_high)

# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

selected = pd.concat(
    [
        moderate,
        selected_high,
    ],
    ignore_index=True,
)

selected["confidence_group"] = [
    "moderate_low" if p >= 0.20 and p <= 0.80
    else "high_confidence"
    for p in selected["probability"]
]

selected = selected.sort_values(
    ["confidence_group", "id"]
).reset_index(drop=True)

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

assert len(moderate) == 5, (
    f"Expected exactly 5 moderate/low cases, got {len(moderate)}"
)

assert len(selected_high) == 5, (
    f"Expected exactly 5 high-confidence controls, got {len(selected_high)}"
)

assert len(selected) == 10
assert selected["id"].is_unique

assert (
    selected["confidence_group"].value_counts()["moderate_low"] == 5
)

assert (
    selected["confidence_group"].value_counts()["high_confidence"] == 5
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(OUTPUT, index=False)

print("=" * 60)
print("E4c-H CASE SELECTION")
print("=" * 60)

print(f"Moderate/low cases: {len(moderate)}")
print(f"High-confidence controls: {len(selected_high)}")
print(f"Total selected originals: {len(selected)}")

print("\nSelected cases:")
print(
    selected[
        [
            "id",
            "domain",
            "label",
            "probability",
            "prediction",
            "correct",
            "confidence_group",
        ]
    ].to_string(index=False)
)

print("\nOutput:")
print(OUTPUT)

print("\nSELECTION VALIDATION PASSED")
