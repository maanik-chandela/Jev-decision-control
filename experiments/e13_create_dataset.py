from pathlib import Path
import pandas as pd


OUTPUT = Path("experiments/data/e13_decision_type_dataset.csv")


CASES = [
    # ============================================================
    # MATHEMATICS — 12 cases
    # 4 easy, 4 moderate, 4 hard
    # ============================================================

    {
        "case_id": "E13_M01",
        "domain": "mathematics",
        "difficulty": "easy",
        "label": 1,
        "state": "17 is a prime number.",
    },
    {
        "case_id": "E13_M02",
        "domain": "mathematics",
        "difficulty": "easy",
        "label": 0,
        "state": "21 is a prime number.",
    },
    {
        "case_id": "E13_M03",
        "domain": "mathematics",
        "difficulty": "easy",
        "label": 1,
        "state": "12 multiplied by 8 equals 96.",
    },
    {
        "case_id": "E13_M04",
        "domain": "mathematics",
        "difficulty": "easy",
        "label": 0,
        "state": "7 multiplied by 9 equals 54.",
    },

    {
        "case_id": "E13_M05",
        "domain": "mathematics",
        "difficulty": "moderate",
        "label": 1,
        "state": "The arithmetic mean of 6, 10, and 14 is 10.",
    },
    {
        "case_id": "E13_M06",
        "domain": "mathematics",
        "difficulty": "moderate",
        "label": 0,
        "state": "The median of 3, 8, 11, 15, and 20 is 8.",
    },
    {
        "case_id": "E13_M07",
        "domain": "mathematics",
        "difficulty": "moderate",
        "label": 1,
        "state": "The sum of the first five positive integers is 15.",
    },
    {
        "case_id": "E13_M08",
        "domain": "mathematics",
        "difficulty": "moderate",
        "label": 0,
        "state": "The positive square root of 81 is 8.",
    },

    {
        "case_id": "E13_M09",
        "domain": "mathematics",
        "difficulty": "hard",
        "label": 1,
        "state": "The 20th term of the arithmetic sequence 4, 7, 10, 13, ... is 61.",
    },
    {
        "case_id": "E13_M10",
        "domain": "mathematics",
        "difficulty": "hard",
        "label": 0,
        "state": "The roots of x^2 - 5x + 6 = 0 are 2 and 4.",
    },
    {
        "case_id": "E13_M11",
        "domain": "mathematics",
        "difficulty": "hard",
        "label": 1,
        "state": "For f(x) = 2x + 3, the value of f(5) is 13.",
    },
    {
        "case_id": "E13_M12",
        "domain": "mathematics",
        "difficulty": "hard",
        "label": 0,
        "state": "For every real number x, x^2 is greater than or equal to x.",
    },

    # ============================================================
    # PROBABILITY — 12 cases
    # ============================================================

    {
        "case_id": "E13_P01",
        "domain": "probability",
        "difficulty": "easy",
        "label": 1,
        "state": "For a fair coin, the probability of obtaining heads is 0.5.",
    },
    {
        "case_id": "E13_P02",
        "domain": "probability",
        "difficulty": "easy",
        "label": 0,
        "state": "For a fair six-sided die, the probability of obtaining an even number is 1/3.",
    },
    {
        "case_id": "E13_P03",
        "domain": "probability",
        "difficulty": "easy",
        "label": 1,
        "state": "If two events are mutually exclusive with probabilities 0.3 and 0.2, their union has probability 0.5.",
    },
    {
        "case_id": "E13_P04",
        "domain": "probability",
        "difficulty": "easy",
        "label": 0,
        "state": "If two independent events have probabilities 0.5 and 0.4, their intersection has probability 0.9.",
    },

    {
        "case_id": "E13_P05",
        "domain": "probability",
        "difficulty": "moderate",
        "label": 1,
        "state": "The probability of obtaining at least one head in two fair coin tosses is 0.75.",
    },
    {
        "case_id": "E13_P06",
        "domain": "probability",
        "difficulty": "moderate",
        "label": 0,
        "state": "The probability of drawing an ace from a standard 52-card deck is 0.25.",
    },
    {
        "case_id": "E13_P07",
        "domain": "probability",
        "difficulty": "moderate",
        "label": 1,
        "state": "If independent events A and B have probabilities 0.5 and 0.4, then P(A intersection B) = 0.2.",
    },
    {
        "case_id": "E13_P08",
        "domain": "probability",
        "difficulty": "moderate",
        "label": 0,
        "state": "If independent events A and B have probabilities 0.6 and 0.5, then P(A intersection B) = 0.35.",
    },

    {
        "case_id": "E13_P09",
        "domain": "probability",
        "difficulty": "hard",
        "label": 1,
        "state": "The probability of obtaining exactly two heads in four fair coin tosses is 0.375.",
    },
    {
        "case_id": "E13_P10",
        "domain": "probability",
        "difficulty": "hard",
        "label": 0,
        "state": "The probability that two fair six-sided dice sum to 7 is 1/12.",
    },
    {
        "case_id": "E13_P11",
        "domain": "probability",
        "difficulty": "hard",
        "label": 1,
        "state": "If P(A) = 0.6, P(B) = 0.5, and P(A intersection B) = 0.2, then P(A union B) = 0.9.",
    },
    {
        "case_id": "E13_P12",
        "domain": "probability",
        "difficulty": "hard",
        "label": 0,
        "state": "For a Bernoulli random variable with parameter p, the variance is p^2(1-p)^2.",
    },

    # ============================================================
    # FORMAL LOGIC — 12 cases
    # ============================================================

    {
        "case_id": "E13_L01",
        "domain": "formal_logic",
        "difficulty": "easy",
        "label": 1,
        "state": "If all mammals are animals and all dogs are mammals, then all dogs are animals.",
    },
    {
        "case_id": "E13_L02",
        "domain": "formal_logic",
        "difficulty": "easy",
        "label": 0,
        "state": "If all birds are animals and all fish are animals, then all birds are fish.",
    },
    {
        "case_id": "E13_L03",
        "domain": "formal_logic",
        "difficulty": "easy",
        "label": 1,
        "state": "If all squares are rectangles, then any object known to be a square is a rectangle.",
    },
    {
        "case_id": "E13_L04",
        "domain": "formal_logic",
        "difficulty": "easy",
        "label": 0,
        "state": "If no cats are reptiles and all tigers are cats, then all tigers are reptiles.",
    },

    {
        "case_id": "E13_L05",
        "domain": "formal_logic",
        "difficulty": "moderate",
        "label": 1,
        "state": "If A implies B and B implies C, then A implies C.",
    },
    {
        "case_id": "E13_L06",
        "domain": "formal_logic",
        "difficulty": "moderate",
        "label": 0,
        "state": "If A implies B and B implies C, then C implies A.",
    },
    {
        "case_id": "E13_L07",
        "domain": "formal_logic",
        "difficulty": "moderate",
        "label": 1,
        "state": "If no A is B and every C is A, then no C is B.",
    },
    {
        "case_id": "E13_L08",
        "domain": "formal_logic",
        "difficulty": "moderate",
        "label": 0,
        "state": "If every A is B and some B is C, then some A is C.",
    },

    {
        "case_id": "E13_L09",
        "domain": "formal_logic",
        "difficulty": "hard",
        "label": 1,
        "state": "If A is logically equivalent to B and B is logically equivalent to C, then A is logically equivalent to C.",
    },
    {
        "case_id": "E13_L10",
        "domain": "formal_logic",
        "difficulty": "hard",
        "label": 0,
        "state": "If A implies B and C implies B, then A implies C.",
    },
    {
        "case_id": "E13_L11",
        "domain": "formal_logic",
        "difficulty": "hard",
        "label": 1,
        "state": "If all poets are writers and no writers are robots, then no poets are robots.",
    },
    {
        "case_id": "E13_L12",
        "domain": "formal_logic",
        "difficulty": "hard",
        "label": 0,
        "state": "If some A is B and some B is C, then some A is C.",
    },

    # ============================================================
    # SCIENCE REASONING — 12 cases
    # ============================================================

    {
        "case_id": "E13_S01",
        "domain": "science_reasoning",
        "difficulty": "easy",
        "label": 1,
        "state": "At standard atmospheric pressure, pure water freezes at approximately 0 degrees Celsius.",
    },
    {
        "case_id": "E13_S02",
        "domain": "science_reasoning",
        "difficulty": "easy",
        "label": 0,
        "state": "A healthy human normally has three lungs.",
    },
    {
        "case_id": "E13_S03",
        "domain": "science_reasoning",
        "difficulty": "easy",
        "label": 1,
        "state": "Light travels faster than sound in air.",
    },
    {
        "case_id": "E13_S04",
        "domain": "science_reasoning",
        "difficulty": "easy",
        "label": 0,
        "state": "Plants obtain most of the carbon dioxide they use for photosynthesis from the soil.",
    },

    {
        "case_id": "E13_S05",
        "domain": "science_reasoning",
        "difficulty": "moderate",
        "label": 1,
        "state": "For an ideal gas, increasing temperature increases the average kinetic energy of its particles.",
    },
    {
        "case_id": "E13_S06",
        "domain": "science_reasoning",
        "difficulty": "moderate",
        "label": 0,
        "state": "Earth's seasons are caused primarily by changes in the Earth's distance from the Sun.",
    },
    {
        "case_id": "E13_S07",
        "domain": "science_reasoning",
        "difficulty": "moderate",
        "label": 1,
        "state": "Antibiotics do not directly treat viral infections.",
    },
    {
        "case_id": "E13_S08",
        "domain": "science_reasoning",
        "difficulty": "moderate",
        "label": 0,
        "state": "All metals are strongly attracted to magnets.",
    },

    {
        "case_id": "E13_S09",
        "domain": "science_reasoning",
        "difficulty": "hard",
        "label": 1,
        "state": "In vacuum, electromagnetic waves propagate at the same speed regardless of their frequency.",
    },
    {
        "case_id": "E13_S10",
        "domain": "science_reasoning",
        "difficulty": "hard",
        "label": 0,
        "state": "In normal DNA, uracil is the standard pyrimidine that replaces thymine.",
    },
    {
        "case_id": "E13_S11",
        "domain": "science_reasoning",
        "difficulty": "hard",
        "label": 1,
        "state": "Newton's third-law force pair consists of forces equal in magnitude and opposite in direction that act on different bodies.",
    },
    {
        "case_id": "E13_S12",
        "domain": "science_reasoning",
        "difficulty": "hard",
        "label": 0,
        "state": "In every physical medium, increasing the frequency of a wave always increases its propagation speed.",
    },

    # ============================================================
    # DATA REASONING — 12 cases
    # ============================================================

    {
        "case_id": "E13_D01",
        "domain": "data_reasoning",
        "difficulty": "easy",
        "label": 1,
        "state": "The arithmetic mean of the values 2, 4, 6, and 8 is 5.",
    },
    {
        "case_id": "E13_D02",
        "domain": "data_reasoning",
        "difficulty": "easy",
        "label": 0,
        "state": "The median of the values 2, 4, 6, and 8 is 6.",
    },
    {
        "case_id": "E13_D03",
        "domain": "data_reasoning",
        "difficulty": "easy",
        "label": 1,
        "state": "A dataset contains 40 successes out of 50 observations. The success rate is 80 percent.",
    },
    {
        "case_id": "E13_D04",
        "domain": "data_reasoning",
        "difficulty": "easy",
        "label": 0,
        "state": "A dataset contains 15 successes out of 20 observations. The success rate is 60 percent.",
    },

    {
        "case_id": "E13_D05",
        "domain": "data_reasoning",
        "difficulty": "moderate",
        "label": 1,
        "state": "Group A has 30 successes out of 40 observations and Group B has 20 successes out of 40 observations. Group A has the higher success rate.",
    },
    {
        "case_id": "E13_D06",
        "domain": "data_reasoning",
        "difficulty": "moderate",
        "label": 0,
        "state": "The median of the values 3, 3, 5, 9, and 10 is 9.",
    },
    {
        "case_id": "E13_D07",
        "domain": "data_reasoning",
        "difficulty": "moderate",
        "label": 1,
        "state": "The mean of the values 12, 15, 18, and 21 is 16.5.",
    },
    {
        "case_id": "E13_D08",
        "domain": "data_reasoning",
        "difficulty": "moderate",
        "label": 0,
        "state": "If a quantity of 80 increases by 25 percent, the resulting quantity is 90.",
    },

    {
        "case_id": "E13_D09",
        "domain": "data_reasoning",
        "difficulty": "hard",
        "label": 1,
        "state": "A classifier has TP = 45, FP = 5, and FN = 5. Its precision and recall are both 0.90.",
    },
    {
        "case_id": "E13_D10",
        "domain": "data_reasoning",
        "difficulty": "hard",
        "label": 0,
        "state": "A classifier has TP = 80, FP = 20, and FN = 10. Its recall is 0.80.",
    },
    {
        "case_id": "E13_D11",
        "domain": "data_reasoning",
        "difficulty": "hard",
        "label": 1,
        "state": "The median of the six values 4, 7, 7, 9, 12, and 15 is 8.",
    },
    {
        "case_id": "E13_D12",
        "domain": "data_reasoning",
        "difficulty": "hard",
        "label": 0,
        "state": "If the Pearson correlation between two variables is zero, this proves that the variables are statistically independent.",
    },
]


# Same underlying state, three different decision representations.
NOUL_INSTRUCTIONS = (
    "Determine whether the statement in the supplied state is true."
)

CHOICE_INSTRUCTIONS = (
    "Classify the truth status of the statement in the supplied state."
)

CHOICE_CRITERIA = {
    "true": "The statement is true.",
    "false": "The statement is false.",
    "uncertain": "The truth value cannot be determined from the supplied state."
}

SCORE_INSTRUCTIONS = (
    "Rate the truth status of the statement in the supplied state."
)

SCORE_CRITERIA = [
    "The statement is false.",
    "The truth value cannot be determined from the supplied state.",
    "The statement is true.",
]


def validate_cases(df):
    assert len(df) == 60, f"Expected 60 cases, got {len(df)}"
    assert df["case_id"].is_unique, "Duplicate case IDs found"
    assert df["state"].is_unique, "Duplicate states found"

    # Overall label balance
    assert df["label"].sum() == 30, "Expected 30 true cases"
    assert (df["label"] == 0).sum() == 30, "Expected 30 false cases"

    # Domain balance
    domain_counts = df["domain"].value_counts()
    expected_domains = {
        "mathematics": 12,
        "probability": 12,
        "formal_logic": 12,
        "science_reasoning": 12,
        "data_reasoning": 12,
    }

    assert domain_counts.to_dict() == expected_domains, (
        f"Unexpected domain counts:\n{domain_counts}"
    )

    # Difficulty balance
    difficulty_counts = df["difficulty"].value_counts()
    expected_difficulty = {
        "easy": 20,
        "moderate": 20,
        "hard": 20,
    }

    assert difficulty_counts.to_dict() == expected_difficulty, (
        f"Unexpected difficulty counts:\n{difficulty_counts}"
    )

    # Every domain should have 4 cases at each difficulty
    cross = pd.crosstab(df["domain"], df["difficulty"])

    assert (cross == 4).all().all(), (
        f"Expected 4 cases per domain/difficulty combination:\n{cross}"
    )

    # Every domain should have 6 true / 6 false
    domain_labels = pd.crosstab(df["domain"], df["label"])

    assert (domain_labels[0] == 6).all()
    assert (domain_labels[1] == 6).all()

    # Required columns
    required = {
        "case_id",
        "domain",
        "difficulty",
        "label",
        "state",
        "noul_instructions",
        "choice_instructions",
        "choice_true",
        "choice_false",
        "choice_uncertain",
        "score_instructions",
        "score_0",
        "score_1",
        "score_2",
    }

    assert required.issubset(df.columns), (
        f"Missing columns: {required - set(df.columns)}"
    )

    print("\nVALIDATION PASSED")
    print("=" * 60)
    print(f"Total cases: {len(df)}")
    print(f"True: {(df['label'] == 1).sum()}")
    print(f"False: {(df['label'] == 0).sum()}")

    print("\nDomain counts:")
    print(domain_counts.sort_index())

    print("\nDifficulty counts:")
    print(difficulty_counts.sort_index())

    print("\nDomain × difficulty:")
    print(cross)

    print("\nDomain × label:")
    print(domain_labels)

    print("\nAll 60 cases:")
    print(
        df[
            ["case_id", "domain", "difficulty", "label", "state"]
        ].to_string(index=False)
    )


def main():
    rows = []

    for case in CASES:
        rows.append({
            "case_id": case["case_id"],
            "domain": case["domain"],
            "difficulty": case["difficulty"],
            "label": case["label"],
            "state": case["state"],

            "noul_instructions": NOUL_INSTRUCTIONS,

            "choice_instructions": CHOICE_INSTRUCTIONS,
            "choice_true": CHOICE_CRITERIA["true"],
            "choice_false": CHOICE_CRITERIA["false"],
            "choice_uncertain": CHOICE_CRITERIA["uncertain"],

            "score_instructions": SCORE_INSTRUCTIONS,
            "score_0": SCORE_CRITERIA[0],
            "score_1": SCORE_CRITERIA[1],
            "score_2": SCORE_CRITERIA[2],
        })

    df = pd.DataFrame(rows)

    validate_cases(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
