import pandas as pd
from pathlib import Path

INPUT = Path("experiments/data/e4c_h_selected_cases.csv")
OUTPUT = Path("experiments/data/e4c_h_counterfactuals.csv")

df = pd.read_csv(INPUT)

rows = []


def add(original, suffix, perturbation, state, question):
    rows.append({
        "counterfactual_id": f"{original['id']}_{suffix}",
        "original_id": original["id"],
        "domain": original["domain"],
        "confidence_group": original["confidence_group"],
        "original_probability": original["probability"],
        "label": int(original["label"]),
        "perturbation_type": perturbation,
        "state": state,
        "question": question,
    })


for _, r in df.iterrows():

    oid = r["id"]

    # ========================================================
    # E4CH_D01
    # ========================================================
    if oid == "E4CH_D01":

        add(
            r, "V1", "paraphrase",
            "The dataset consists of the values 2, 4, 6, 8, and 10.",
            "Is the arithmetic mean of these values 6?"
        )

        add(
            r, "V2", "question_form",
            "A dataset contains the five values 2, 4, 6, 8, and 10.",
            "Does the mean of the dataset equal 6?"
        )

        add(
            r, "V3", "state_rewrite",
            "The observed values in a dataset are 2, 4, 6, 8, and 10.",
            "Is their average equal to 6?"
        )

    # ========================================================
    # E4CH_L01
    # ========================================================
    elif oid == "E4CH_L01":

        add(
            r, "V1", "paraphrase",
            "Every member of set A is also a member of set B, "
            "and every member of set B is also a member of set C.",
            "Is every member of A necessarily a member of C?"
        )

        add(
            r, "V2", "question_form",
            "All A objects are B objects, and all B objects are C objects.",
            "Does belonging to A necessarily imply belonging to C?"
        )

        add(
            r, "V3", "state_rewrite",
            "The inclusion relationships are A ⊆ B and B ⊆ C.",
            "Must A be a subset of C?"
        )

    # ========================================================
    # E4CH_M01
    # ========================================================
    elif oid == "E4CH_M01":

        add(
            r, "V1", "paraphrase",
            "Consider the linear equation 3x + 7 = 25. "
            "One proposed solution is x = 6.",
            "Is x = 6 a solution of the equation?"
        )

        add(
            r, "V2", "question_form",
            "The equation is 3x + 7 = 25.",
            "Does substituting x = 6 satisfy the equation?"
        )

        add(
            r, "V3", "state_rewrite",
            "For the equation 3x + 7 = 25, evaluate the proposed value x = 6.",
            "Is the proposed value a valid solution?"
        )

    # ========================================================
    # E4CH_P02
    # ========================================================
    elif oid == "E4CH_P02":

        add(
            r, "V1", "paraphrase",
            "A fair coin is tossed repeatedly, with each toss independent "
            "of the previous tosses.",
            "Is the probability that the first two tosses are both heads 1/4?"
        )

        add(
            r, "V2", "question_form",
            "Consider repeated independent tosses of a fair coin.",
            "Would the first two tosses both being heads have probability 1/4?"
        )

        add(
            r, "V3", "state_rewrite",
            "Each toss of a fair coin has probability 1/2 of producing heads, "
            "and the tosses are independent.",
            "Is P(first toss = heads AND second toss = heads) equal to 1/4?"
        )

    # ========================================================
    # E4CH_S01
    # ========================================================
    elif oid == "E4CH_S01":

        add(
            r, "V1", "paraphrase",
            "An ideal gas is heated from 300 K to 600 K while its pressure "
            "is held constant.",
            "According to Charles's law, does its volume double?"
        )

        add(
            r, "V2", "question_form",
            "The pressure of an ideal gas remains constant as its temperature "
            "increases from 300 K to 600 K.",
            "Does the gas volume become twice its initial value?"
        )

        add(
            r, "V3", "state_rewrite",
            "An ideal gas undergoes a temperature change from 300 K to 600 K "
            "with pressure unchanged.",
            "Is the final volume 2 times the initial volume?"
        )

    # ========================================================
    # E4CH_L10
    # ========================================================
    elif oid == "E4CH_L10":

        add(
            r, "V1", "paraphrase",
            "Some students study mathematics, and every student who studies "
            "mathematics also studies algebra.",
            "Must at least one student study algebra?"
        )

        add(
            r, "V2", "question_form",
            "At least one student is a mathematics student. "
            "All mathematics students study algebra.",
            "Does it follow that at least one student studies algebra?"
        )

        add(
            r, "V3", "state_rewrite",
            "There exists at least one mathematics student, and the set of "
            "mathematics students is contained within the set of algebra students.",
            "Must the group contain an algebra student?"
        )

    # ========================================================
    # E4CH_L11
    # ========================================================
    elif oid == "E4CH_L11":

        add(
            r, "V1", "paraphrase",
            "No mammal lays eggs. The platypus is classified as a mammal.",
            "Does it follow that the platypus does not lay eggs?"
        )

        add(
            r, "V2", "question_form",
            "The platypus is a mammal, and mammals do not lay eggs.",
            "Must the platypus be unable to lay eggs?"
        )

        add(
            r, "V3", "state_rewrite",
            "Membership in the mammal category implies not laying eggs. "
            "The platypus belongs to the mammal category.",
            "Can the platypus lay eggs under these stated premises?"
        )

    # ========================================================
    # E4CH_L15
    # ========================================================
    elif oid == "E4CH_L15":

        add(
            r, "V1", "paraphrase",
            "Whenever the server is online, the website responds. "
            "The website is currently not responding.",
            "Must the server therefore be offline?"
        )

        add(
            r, "V2", "question_form",
            "If the server is online, the website responds. "
            "The website does not respond.",
            "Does the evidence logically imply that the server is offline?"
        )

        add(
            r, "V3", "state_rewrite",
            "Online server status is sufficient for the website to respond. "
            "The website is observed not to respond.",
            "Is server-offline status necessarily implied?"
        )

    # ========================================================
    # E4CH_P01
    # ========================================================
    elif oid == "E4CH_P01":

        add(
            r, "V1", "paraphrase",
            "A box holds 3 red balls, 4 blue balls, and 5 green balls. "
            "Two balls are selected without replacement.",
            "Is the probability that both selected balls are red equal to 1/22?"
        )

        add(
            r, "V2", "question_form",
            "There are 3 red balls among 12 total balls, and two balls are drawn "
            "without replacement.",
            "Is P(both draws are red) equal to 1/22?"
        )

        add(
            r, "V3", "state_rewrite",
            "The urn contains 12 balls in total, exactly 3 of which are red. "
            "Two balls are sampled sequentially without replacement.",
            "Does the probability of two red draws equal 1/22?"
        )

    # ========================================================
    # E4CH_P09
    # ========================================================
    elif oid == "E4CH_P09":

        add(
            r, "V1", "paraphrase",
            "A quantity changes from an initial value of 80 to a final value of 100.",
            "Is the percentage increase exactly 20%?"
        )

        add(
            r, "V2", "question_form",
            "The original quantity is 80 and the new quantity is 100.",
            "Does the change from 80 to 100 represent a 20% increase?"
        )

        add(
            r, "V3", "state_rewrite",
            "The quantity increases by 20 units, from 80 units to 100 units.",
            "Is the relative increase equal to 20%?"
        )


# ============================================================
# VALIDATION
# ============================================================

out = pd.DataFrame(rows)

assert len(out) == 30
assert out["counterfactual_id"].is_unique
assert out["original_id"].nunique() == 10

counts = out["perturbation_type"].value_counts()

assert counts["paraphrase"] == 10
assert counts["question_form"] == 10
assert counts["state_rewrite"] == 10

assert out.groupby("original_id").size().eq(3).all()

assert set(out["original_id"]) == set(df["id"])

# Ground truth must be inherited exactly.
for _, row in out.iterrows():
    original = df[df["id"] == row["original_id"]].iloc[0]
    assert row["label"] == original["label"]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)

print("=" * 60)
print("E4c-H COUNTERFACTUAL DATASET")
print("=" * 60)

print(f"Original cases: {df['id'].nunique()}")
print(f"Counterfactual evaluations: {len(out)}")

print("\nPerturbation counts:")
print(counts)

print("\nConfidence groups:")
print(out["confidence_group"].value_counts())

print("\nSelected originals:")
print(
    df[
        [
            "id",
            "domain",
            "probability",
            "label",
            "confidence_group",
        ]
    ].to_string(index=False)
)

print("\nVALIDATION PASSED")
print(f"Output: {OUTPUT}")

