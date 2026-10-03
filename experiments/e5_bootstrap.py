import numpy as np
import pandas as pd

JOINED = "experiments/data/e5_joined_results.csv"
N_BOOT = 10000
SEED = 42

df = pd.read_csv(JOINED)

# ------------------------------------------------------------
# Independent unit = original case.
# Each case must have exactly 3 counterfactual variants.
# ------------------------------------------------------------

case_ids = df["original_id"].unique()
n_cases = len(case_ids)

if n_cases != 60:
    raise ValueError(f"Expected 60 cases, found {n_cases}")

variants_per_case = df.groupby("original_id").size()

if not (variants_per_case == 3).all():
    raise ValueError("Every original case must have exactly 3 variants.")

print("E5 CASE-LEVEL BOOTSTRAP")
print("=" * 60)
print(f"Independent cases: {n_cases}")
print(f"Variants per case: {variants_per_case.unique().tolist()}")
print(f"Bootstrap samples: {N_BOOT}")
print()

# ------------------------------------------------------------
# Sort so each case occupies exactly 3 consecutive rows.
# ------------------------------------------------------------

df = df.sort_values(
    ["original_id", "counterfactual_id"]
).reset_index(drop=True)

# Verify structure.
for case_id, group in df.groupby("original_id"):
    if len(group) != 3:
        raise ValueError(
            f"{case_id} has {len(group)} variants instead of 3."
        )

# ------------------------------------------------------------
# Case-level metrics.
# ------------------------------------------------------------

case = (
    df.groupby("original_id")
    .agg(
        domain=("domain", "first"),
        mean_abs_delta_p=("abs_delta_p", "mean"),
        mean_confidence_loss=("confidence_loss", "mean"),
        decision_changes=("decision_changed", "sum"),
        adversarial_errors=("adversarial_error", "sum"),
    )
    .reindex(case_ids)
    .reset_index()
)

# ------------------------------------------------------------
# Reshape variant probabilities into:
#
# 60 cases × 3 variants
#
# Because the dataframe is sorted and every case has 3 rows,
# this is safe.
# ------------------------------------------------------------

delta_values = df["abs_delta_p"].to_numpy()

if len(delta_values) != 180:
    raise ValueError(
        f"Expected 180 counterfactuals, found {len(delta_values)}"
    )

delta_matrix = delta_values.reshape(n_cases, 3)

print(f"Delta matrix shape: {delta_matrix.shape}")
print()

# ------------------------------------------------------------
# Bootstrap
#
# Resample the 60 original cases.
# Each sampled case carries all 3 variants.
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

error_rates = np.empty(N_BOOT)
decision_rates = np.empty(N_BOOT)
mean_abs_deltas = np.empty(N_BOOT)
median_abs_deltas = np.empty(N_BOOT)
confidence_losses = np.empty(N_BOOT)

for i in range(N_BOOT):

    # Sample 60 original cases with replacement.
    idx = rng.integers(0, n_cases, size=n_cases)

    sampled_case = case.iloc[idx]

    # Every case has 3 variants.
    total_variants = n_cases * 3

    error_rates[i] = (
        sampled_case["adversarial_errors"].sum()
        / total_variants
    )

    decision_rates[i] = (
        sampled_case["decision_changes"].sum()
        / total_variants
    )

    mean_abs_deltas[i] = (
        sampled_case["mean_abs_delta_p"].mean()
    )

    confidence_losses[i] = (
        sampled_case["mean_confidence_loss"].mean()
    )

    # Keep all three variants together.
    sampled_deltas = delta_matrix[idx].reshape(-1)

    median_abs_deltas[i] = np.median(sampled_deltas)


def summarize(name, values):
    estimate = np.mean(values)
    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)

    print(
        f"{name:27s}: "
        f"{estimate:.4f} "
        f"[{lower:.4f}, {upper:.4f}]"
    )

    return {
        "metric": name,
        "estimate": estimate,
        "ci_lower_95": lower,
        "ci_upper_95": upper,
    }


print("95% BOOTSTRAP CONFIDENCE INTERVALS")
print("-" * 60)

rows = []

rows.append(
    summarize(
        "Adversarial error rate",
        error_rates,
    )
)

rows.append(
    summarize(
        "Decision-change rate",
        decision_rates,
    )
)

rows.append(
    summarize(
        "Mean |delta p|",
        mean_abs_deltas,
    )
)

rows.append(
    summarize(
        "Median |delta p|",
        median_abs_deltas,
    )
)

rows.append(
    summarize(
        "Mean confidence loss",
        confidence_losses,
    )

)

# ------------------------------------------------------------
# Save results.
# ------------------------------------------------------------

out = pd.DataFrame(rows)

output = "experiments/data/e5_bootstrap_summary.csv"

out.to_csv(output, index=False)

case.to_csv(
    "experiments/data/e5_case_level_bootstrap_input.csv",
    index=False,
)

print()
print(f"Saved: {output}")
print(
    "Saved: experiments/data/"
    "e5_case_level_bootstrap_input.csv"
)
