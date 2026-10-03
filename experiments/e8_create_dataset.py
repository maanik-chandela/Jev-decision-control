import pandas as pd
from pathlib import Path

OUT = Path("experiments/data/e8_decision_representation_dataset.csv")

# 60 underlying propositions:
# 12 per domain, balanced 30 True / 30 False.
cases = [
    # ---------------- MATHEMATICS ----------------
    ("M01", "mathematics", 1, "A square has side length 6 cm. Its area is 36 square cm."),
    ("M02", "mathematics", 0, "A rectangle has length 8 cm and width 3 cm. Its area is 30 square cm."),
    ("M03", "mathematics", 1, "The sum of the interior angles of a triangle is 180 degrees."),
    ("M04", "mathematics", 0, "The average of 4, 8, and 12 is 10."),
    ("M05", "mathematics", 1, "If x = 5, then 2x + 3 = 13."),
    ("M06", "mathematics", 0, "The square root of 81 is 8."),
    ("M07", "mathematics", 1, "A prime number greater than 2 is odd."),
    ("M08", "mathematics", 0, "The perimeter of a square with side 5 cm is 15 cm."),
    ("M09", "mathematics", 1, "If a number is divisible by 10, it is divisible by 5."),
    ("M10", "mathematics", 0, "The fraction 3/4 is equal to 2/3."),
    ("M11", "mathematics", 1, "The derivative of x^2 with respect to x is 2x."),
    ("M12", "mathematics", 0, "A right triangle with legs 3 and 4 has hypotenuse 6."),

    # ---------------- PROBABILITY ----------------
    ("P01", "probability", 1, "For a fair six-sided die, the probability of rolling an even number is 1/2."),
    ("P02", "probability", 0, "For a fair six-sided die, the probability of rolling a number greater than 4 is 1/2."),
    ("P03", "probability", 1, "An impossible event has probability 0."),
    ("P04", "probability", 0, "A certain event has probability 0."),
    ("P05", "probability", 1, "The probabilities of all outcomes in a finite sample space sum to 1."),
    ("P06", "probability", 0, "A probability can be greater than 1."),
    ("P07", "probability", 1, "For independent events A and B, P(A and B) equals P(A)P(B)."),
    ("P08", "probability", 0, "If P(A) = 0.3, then P(not A) = 0.3."),
    ("P09", "probability", 1, "In a finite sample space where every elementary outcome has positive probability, an event with probability 0 is impossible."),
    ("P10", "probability", 0, "For a fair coin, the probability of heads is 0.7."),
    ("P11", "probability", 1, "The probability of an event and its complement sum to 1."),
    ("P12", "probability", 0, "Two mutually exclusive events must always be independent."),

    # ---------------- FORMAL LOGIC ----------------
    ("L01", "formal_logic", 1, "If A implies B and A is true, then B must be true."),
    ("L02", "formal_logic", 0, "If A implies B and B is true, then A must be true."),
    ("L03", "formal_logic", 1, "If A implies B and B is false, then A must be false."),
    ("L04", "formal_logic", 0, "If A implies B and A is false, then B must be false."),
    ("L05", "formal_logic", 1, "If all cats are mammals and Luna is a cat, then Luna is a mammal."),
    ("L06", "formal_logic", 0, "If all cats are mammals and Luna is a mammal, then Luna must be a cat."),
    ("L07", "formal_logic", 1, "If no birds are mammals and all sparrows are birds, then no sparrow is a mammal."),
    ("L08", "formal_logic", 0, "If some A are B, then every A must be B."),
    ("L09", "formal_logic", 1, "If every A is B and every B is C, then every A is C."),
    ("L10", "formal_logic", 0, "If no A is B, then some A must be B."),
    ("L11", "formal_logic", 1, "A statement and its logical negation cannot both be true."),
    ("L12", "formal_logic", 0, "If A or B is true, then A and B must both be true."),

    # ---------------- SCIENCE REASONING ----------------
    ("S01", "science_reasoning", 1, "Water freezes at 0 degrees Celsius under standard atmospheric pressure."),
    ("S02", "science_reasoning", 0, "Water boils at 50 degrees Celsius under standard atmospheric pressure."),
    ("S03", "science_reasoning", 1, "Plants use light energy during photosynthesis."),
    ("S04", "science_reasoning", 0, "Sound travels faster through a vacuum than through air."),
    ("S05", "science_reasoning", 1, "The Earth completes approximately one revolution around the Sun in one year."),
    ("S06", "science_reasoning", 0, "The Moon produces more visible light than the Sun."),
    ("S07", "science_reasoning", 1, "Increasing mass while keeping force constant decreases acceleration."),
    ("S08", "science_reasoning", 0, "Objects with greater mass always fall faster than lighter objects in a vacuum."),
    ("S09", "science_reasoning", 1, "Carbon dioxide is a greenhouse gas."),
    ("S10", "science_reasoning", 0, "Electrons have a positive electric charge."),
    ("S11", "science_reasoning", 1, "A balanced chemical equation conserves the number of atoms of each element."),
    ("S12", "science_reasoning", 0, "All metals are gases at room temperature."),

    # ---------------- DATA REASONING ----------------
    ("D01", "data_reasoning", 1, "A dataset containing 20 observations has 5 positive observations, so the positive proportion is 25%."),
    ("D02", "data_reasoning", 0, "A dataset containing 20 observations has 5 positive observations, so the positive proportion is 40%."),
    ("D03", "data_reasoning", 1, "If a classifier makes 90 correct predictions out of 100, its accuracy is 90%."),
    ("D04", "data_reasoning", 0, "If a classifier makes 90 correct predictions out of 100, its accuracy is 95%."),
    ("D05", "data_reasoning", 1, "The median of 2, 4, and 10 is 4."),
    ("D06", "data_reasoning", 0, "The median of 2, 4, and 10 is 5."),
    ("D07", "data_reasoning", 1, "If every value in a dataset increases by 5, the mean also increases by 5."),
    ("D08", "data_reasoning", 0, "If every value in a dataset increases by 5, the mean remains unchanged."),
    ("D09", "data_reasoning", 1, "A precision of 80% means that 80% of predicted positive cases are actually positive."),
    ("D10", "data_reasoning", 0, "A precision of 80% means that 80% of all actual positive cases were detected."),
    ("D11", "data_reasoning", 1, "A dataset with 10 values has 3 positive values, so its positive proportion is 30%."),
    ("D12", "data_reasoning", 0, "A dataset with 10 values has 3 positive values, so its positive proportion is 50%."),
]

assert len(cases) == 60
assert sum(x[2] for x in cases) == 30
assert sum(1 - x[2] for x in cases) == 30

rows = []

for case_id, domain, label, proposition in cases:

    formulations = {
        "direct": f"Is it true that {proposition}",
        "predicate": f"Does the following statement hold: {proposition}",
        "classification": f"Should the following statement be classified as true: {proposition}",
    }

    for representation, question in formulations.items():
        rows.append({
            "case_id": case_id,
            "domain": domain,
            "label": label,
            "proposition": proposition,
            "representation": representation,
            "question": question,
        })

df = pd.DataFrame(rows)

assert len(df) == 180
assert df["case_id"].nunique() == 60
assert df.groupby("case_id").size().eq(3).all()
assert df.groupby("case_id")["label"].nunique().eq(1).all()
assert df.groupby("representation").size().eq(60).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"Saved {len(df)} rows to {OUT}")
print()
print("Cases:", df["case_id"].nunique())
print("Rows:", len(df))
print("Labels:")
print(df.groupby("representation")["label"].value_counts())
print()
print("Domains:")
print(df.groupby(["domain", "representation"]).size())
