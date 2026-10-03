import os
import pandas as pd


OUTPUT_FILE = "experiments/data/e6_domain_shift_dataset.csv"


# ============================================================
# E6 DOMAIN-SHIFT DATASET
#
# 50 matched pairs = 100 total cases.
#
# Each pair preserves the underlying decision structure and
# ground-truth answer while changing the semantic domain.
#
# IMPORTANT:
# This dataset must be audited and frozen BEFORE any JEV calls.
# ============================================================


CASES = [
    # --------------------------------------------------------
    # PAIR 01-10
    # Mathematics -> Finance
    # --------------------------------------------------------

    {
        "core_id": "C01",
        "label": 0,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "A quantity has an initial value of 100. It increases by 20 percent and then decreases by 20 percent.",
        "source_question": "Does the final value equal the original value of 100?",
        "target_state": "An investment account has an initial value of 100 units. Its value increases by 20 percent and then decreases by 20 percent.",
        "target_question": "Does the final account value equal the original value of 100 units?",
    },

    {
        "core_id": "C02",
        "label": 1,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "A rectangle has length 8 units and width 5 units.",
        "source_question": "Is its area 40 square units?",
        "target_state": "A financial portfolio contains 8 units of one asset and 5 units of another, with each unit counted equally for a simple rectangular allocation calculation.",
        "target_question": "Is the corresponding allocation product 40 units?",
    },

    {
        "core_id": "C03",
        "label": 0,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "A number is divided by 4 and the result is 7.",
        "source_question": "Is the original number 24?",
        "target_state": "A company's total revenue divided equally among 4 reporting periods gives 7 units per period.",
        "target_question": "Is the total revenue 24 units?",
    },

    {
        "core_id": "C04",
        "label": 1,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "A price is 80 units. It increases by 25 percent.",
        "source_question": "Is the new price 100 units?",
        "target_state": "A financial asset is priced at 80 units and rises by 25 percent.",
        "target_question": "Is the new asset price 100 units?",
    },

    {
        "core_id": "C05",
        "label": 0,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "A quantity is 60. Twenty percent of the quantity is 12.",
        "source_question": "Is this statement false?",
        "target_state": "A company has 60 units of revenue. Twenty percent of that revenue is 12 units.",
        "target_question": "Is this statement false?",
    },

    {
        "core_id": "C06",
        "label": 1,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "Five equal quantities sum to 45.",
        "source_question": "Is each quantity equal to 9?",
        "target_state": "Five equal business expenses have a combined total of 45 units.",
        "target_question": "Is each expense equal to 9 units?",
    },

    {
        "core_id": "C07",
        "label": 0,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "A square has side length 6 units.",
        "source_question": "Is its perimeter 36 units?",
        "target_state": "A square financial dashboard has six equal intervals on each side.",
        "target_question": "Is the total perimeter count 36 intervals?",
    },

    {
        "core_id": "C08",
        "label": 1,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "The values 4, 6, and 8 are given.",
        "source_question": "Is their arithmetic mean 6?",
        "target_state": "Three equally weighted investments have returns of 4, 6, and 8 percent.",
        "target_question": "Is their arithmetic mean return 6 percent?",
    },

    {
        "core_id": "C09",
        "label": 0,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "A number is 15.",
        "source_question": "Is 30 percent of the number equal to 6?",
        "target_state": "A financial balance is 15 units.",
        "target_question": "Is 30 percent of the balance equal to 6 units?",
    },

    {
        "core_id": "C10",
        "label": 1,
        "source_domain": "mathematics",
        "target_domain": "finance",
        "source_state": "A quantity increases from 50 to 60.",
        "source_question": "Is the percentage increase 20 percent?",
        "target_state": "A company's revenue increases from 50 units to 60 units.",
        "target_question": "Is the percentage increase in revenue 20 percent?",
    },


    # --------------------------------------------------------
    # PAIR 11-20
    # Probability -> Medical
    # --------------------------------------------------------

    {
        "core_id": "C11",
        "label": 1,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "A fair coin is tossed once.",
        "source_question": "Is the probability of obtaining heads 0.5?",
        "target_state": "A diagnostic test has equal probability of producing either of two equally likely outcomes under a specified null condition.",
        "target_question": "Is the probability of the specified positive outcome 0.5?",
    },

    {
        "core_id": "C12",
        "label": 0,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "A fair six-sided die is rolled once.",
        "source_question": "Is the probability of rolling a number greater than 4 equal to 0.5?",
        "target_state": "A medical screening procedure has six equally likely outcome categories under a controlled test.",
        "target_question": "Is the probability of obtaining one of two specified categories equal to 0.5?",
    },

    {
        "core_id": "C13",
        "label": 1,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "A fair coin is tossed twice.",
        "source_question": "Is the probability of obtaining two heads equal to 0.25?",
        "target_state": "A treatment outcome is independently successful on two trials, with probability 0.5 on each trial.",
        "target_question": "Is the probability of success on both trials equal to 0.25?",
    },

    {
        "core_id": "C14",
        "label": 0,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "A fair die is rolled once.",
        "source_question": "Is the probability of rolling an even number equal to 0.25?",
        "target_state": "A controlled medical experiment has four equally likely outcome categories, two of which are classified as even-coded outcomes.",
        "target_question": "Is the probability of obtaining an even-coded outcome 0.25?",
    },

    {
        "core_id": "C15",
        "label": 1,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "An event has probability 0.3.",
        "source_question": "Is the probability that the event does not occur 0.7?",
        "target_state": "A medical event has probability 0.3 under a specified model.",
        "target_question": "Is the probability that the event does not occur 0.7?",
    },

    {
        "core_id": "C16",
        "label": 0,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "Two independent events have probabilities 0.5 and 0.4.",
        "source_question": "Is the probability that both occur equal to 0.5?",
        "target_state": "Two independent clinical outcomes have probabilities 0.5 and 0.4.",
        "target_question": "Is the probability that both outcomes occur equal to 0.5?",
    },

    {
        "core_id": "C17",
        "label": 1,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "A fair die is rolled once.",
        "source_question": "Is the probability of rolling a number less than 3 equal to 1/3?",
        "target_state": "A controlled medical process has six equally likely outcome categories, with two categories satisfying a specified condition.",
        "target_question": "Is the probability of satisfying the specified condition equal to 1/3?",
    },

    {
        "core_id": "C18",
        "label": 0,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "A fair coin is tossed three times.",
        "source_question": "Is the probability of obtaining three heads equal to 0.5?",
        "target_state": "A treatment succeeds independently on three trials, with success probability 0.5 each time.",
        "target_question": "Is the probability of three consecutive successes equal to 0.5?",
    },

    {
        "core_id": "C19",
        "label": 1,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "An event has probability 0.8.",
        "source_question": "Is the probability of the event occurring at least once in one trial 0.8?",
        "target_state": "A medical outcome has probability 0.8 in one trial.",
        "target_question": "Is the probability of observing the outcome in one trial 0.8?",
    },

    {
        "core_id": "C20",
        "label": 0,
        "source_domain": "probability",
        "target_domain": "medical",
        "source_state": "There are 10 equally likely outcomes, and 3 satisfy a specified condition.",
        "source_question": "Is the probability of the condition 0.5?",
        "target_state": "A clinical protocol has 10 equally likely outcome categories, and 3 satisfy a specified criterion.",
        "target_question": "Is the probability of satisfying the criterion 0.5?",
    },


    # --------------------------------------------------------
    # PAIR 21-30
    # Formal logic -> Legal/rule reasoning
    # --------------------------------------------------------

    {
        "core_id": "C21",
        "label": 1,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "If A occurs, then B occurs. A occurs.",
        "source_question": "Must B occur?",
        "target_state": "A rule states that if condition A is satisfied, condition B must follow. Condition A is satisfied.",
        "target_question": "Must condition B follow?",
    },

    {
        "core_id": "C22",
        "label": 0,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "If A occurs, then B occurs. B occurs.",
        "source_question": "Must A have occurred?",
        "target_state": "A rule states that if condition A is satisfied, condition B follows. Condition B is observed.",
        "target_question": "Must condition A have been satisfied?",
    },

    {
        "core_id": "C23",
        "label": 1,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "All members of set A are members of set B. Object x is a member of A.",
        "source_question": "Must x be a member of B?",
        "target_state": "Every person satisfying rule A also satisfies rule B. Person x satisfies rule A.",
        "target_question": "Must person x satisfy rule B?",
    },

    {
        "core_id": "C24",
        "label": 0,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "Some members of A are members of B. Object x is a member of A.",
        "source_question": "Must x be a member of B?",
        "target_state": "Some people satisfying condition A also satisfy condition B. Person x satisfies condition A.",
        "target_question": "Must person x satisfy condition B?",
    },

    {
        "core_id": "C25",
        "label": 1,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "No member of A is a member of B. Object x is a member of A.",
        "source_question": "Must x not be a member of B?",
        "target_state": "No person satisfying condition A may satisfy condition B. Person x satisfies condition A.",
        "target_question": "Must person x fail to satisfy condition B?",
    },

    {
        "core_id": "C26",
        "label": 0,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "If A occurs, then B occurs. A does not occur.",
        "source_question": "Must B not occur?",
        "target_state": "A rule states that A implies B. Condition A is not satisfied.",
        "target_question": "Must condition B therefore be absent?",
    },

    {
        "core_id": "C27",
        "label": 1,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "If A occurs, then B occurs. B does not occur.",
        "source_question": "Can A have occurred?",
        "target_state": "A rule states that A implies B. B is known not to have occurred.",
        "target_question": "Can condition A have been satisfied?",
    },

    {
        "core_id": "C28",
        "label": 0,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "All A are B. All B are C. Object x is not C.",
        "source_question": "Must x be A?",
        "target_state": "Every person satisfying rule A satisfies B, and every person satisfying B satisfies C. Person x does not satisfy C.",
        "target_question": "Must person x satisfy rule A?",
    },

    {
        "core_id": "C29",
        "label": 1,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "If A occurs, then B occurs, and if B occurs, then C occurs. A occurs.",
        "source_question": "Must C occur?",
        "target_state": "Rule A implies rule B, and rule B implies rule C. Condition A is satisfied.",
        "target_question": "Must condition C follow?",
    },

    {
        "core_id": "C30",
        "label": 0,
        "source_domain": "formal_logic",
        "target_domain": "legal_rule_reasoning",
        "source_state": "Some A are B. No B are C.",
        "source_question": "Must every A be outside C?",
        "target_state": "Some people satisfying A also satisfy B. No person satisfying B may satisfy C.",
        "target_question": "Must every person satisfying A fail to satisfy C?",
    },


    # --------------------------------------------------------
    # PAIR 31-40
    # Science reasoning -> Engineering
    # --------------------------------------------------------

    {
        "core_id": "C31",
        "label": 1,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "An object travels 100 meters in 10 seconds at constant speed.",
        "source_question": "Is its speed 10 meters per second?",
        "target_state": "A machine moves a component 100 meters in 10 seconds at constant speed.",
        "target_question": "Is the component's speed 10 meters per second?",
    },

    {
        "core_id": "C32",
        "label": 0,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "An object travels 120 meters in 10 seconds at constant speed.",
        "source_question": "Is its speed 10 meters per second?",
        "target_state": "A machine moves a component 120 meters in 10 seconds at constant speed.",
        "target_question": "Is the component's speed 10 meters per second?",
    },

    {
        "core_id": "C33",
        "label": 1,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "A sample has mass 20 grams and volume 4 cubic centimeters.",
        "source_question": "Is its density 5 grams per cubic centimeter?",
        "target_state": "An engineered material has mass 20 grams and volume 4 cubic centimeters.",
        "target_question": "Is its density 5 grams per cubic centimeter?",
    },

    {
        "core_id": "C34",
        "label": 0,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "A sample has mass 20 grams and volume 5 cubic centimeters.",
        "source_question": "Is its density 5 grams per cubic centimeter?",
        "target_state": "An engineered material has mass 20 grams and volume 5 cubic centimeters.",
        "target_question": "Is its density 5 grams per cubic centimeter?",
    },

    {
        "core_id": "C35",
        "label": 1,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "A process requires 2 units of energy for each unit of output. It produces 10 units of output.",
        "source_question": "Does it require 20 units of energy?",
        "target_state": "A manufacturing process consumes 2 energy units per manufactured unit and produces 10 units.",
        "target_question": "Does it require 20 energy units?",
    },

    {
        "core_id": "C36",
        "label": 0,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "A process requires 3 units of energy for each unit of output. It produces 10 units of output.",
        "source_question": "Does it require 20 units of energy?",
        "target_state": "A manufacturing process consumes 3 energy units per manufactured unit and produces 10 units.",
        "target_question": "Does it require 20 energy units?",
    },

    {
        "core_id": "C37",
        "label": 1,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "A system receives 100 units of input and has efficiency 80 percent.",
        "source_question": "Is its useful output 80 units?",
        "target_state": "An engineered system receives 100 units of input energy and operates at 80 percent efficiency.",
        "target_question": "Is its useful output 80 units?",
    },

    {
        "core_id": "C38",
        "label": 0,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "A system receives 100 units of input and has efficiency 70 percent.",
        "source_question": "Is its useful output 80 units?",
        "target_state": "An engineered system receives 100 units of input energy and operates at 70 percent efficiency.",
        "target_question": "Is its useful output 80 units?",
    },

    {
        "core_id": "C39",
        "label": 1,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "A temperature rises from 20 degrees to 30 degrees.",
        "source_question": "Is the increase 10 degrees?",
        "target_state": "An engineered component's temperature rises from 20 degrees to 30 degrees.",
        "target_question": "Is the temperature increase 10 degrees?",
    },

    {
        "core_id": "C40",
        "label": 0,
        "source_domain": "science_reasoning",
        "target_domain": "engineering",
        "source_state": "A temperature rises from 20 degrees to 35 degrees.",
        "source_question": "Is the increase 10 degrees?",
        "target_state": "An engineered component's temperature rises from 20 degrees to 35 degrees.",
        "target_question": "Is the temperature increase 10 degrees?",
    },


    # --------------------------------------------------------
    # PAIR 41-50
    # Data reasoning -> Business/operations
    # --------------------------------------------------------

    {
        "core_id": "C41",
        "label": 1,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A dataset contains the values 2, 4, 6, 8, and 10.",
        "source_question": "Is the mean equal to 6?",
        "target_state": "Five equally weighted business observations have values 2, 4, 6, 8, and 10.",
        "target_question": "Is the mean equal to 6?",
    },

    {
        "core_id": "C42",
        "label": 0,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A dataset contains the values 2, 4, 6, 8, and 10.",
        "source_question": "Is the median equal to 8?",
        "target_state": "Five ordered business observations have values 2, 4, 6, 8, and 10.",
        "target_question": "Is the median equal to 8?",
    },

    {
        "core_id": "C43",
        "label": 1,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A dataset has 20 observations, of which 5 belong to class A.",
        "source_question": "Does class A represent 25 percent of the observations?",
        "target_state": "A business dataset has 20 transactions, of which 5 belong to category A.",
        "target_question": "Does category A represent 25 percent of the transactions?",
    },

    {
        "core_id": "C44",
        "label": 0,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A dataset has 20 observations, of which 8 belong to class A.",
        "source_question": "Does class A represent 25 percent of the observations?",
        "target_state": "A business dataset has 20 transactions, of which 8 belong to category A.",
        "target_question": "Does category A represent 25 percent of the transactions?",
    },

    {
        "core_id": "C45",
        "label": 1,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A classifier correctly identifies 90 out of 100 cases.",
        "source_question": "Is its accuracy 90 percent?",
        "target_state": "A business screening system correctly processes 90 out of 100 transactions.",
        "target_question": "Is its accuracy rate 90 percent?",
    },

    {
        "core_id": "C46",
        "label": 0,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A classifier correctly identifies 80 out of 100 cases.",
        "source_question": "Is its accuracy 90 percent?",
        "target_state": "A business screening system correctly processes 80 out of 100 transactions.",
        "target_question": "Is its accuracy rate 90 percent?",
    },

    {
        "core_id": "C47",
        "label": 1,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A dataset contains 40 positive cases and 60 negative cases.",
        "source_question": "Do positive cases make up 40 percent of the dataset?",
        "target_state": "A business dataset contains 40 positive outcomes and 60 negative outcomes.",
        "target_question": "Do positive outcomes make up 40 percent of the dataset?",
    },

    {
        "core_id": "C48",
        "label": 0,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A dataset contains 30 positive cases and 70 negative cases.",
        "source_question": "Do positive cases make up 40 percent of the dataset?",
        "target_state": "A business dataset contains 30 positive outcomes and 70 negative outcomes.",
        "target_question": "Do positive outcomes make up 40 percent of the dataset?",
    },

    {
        "core_id": "C49",
        "label": 1,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A dataset contains 50 observations. The average value is 10.",
        "source_question": "Is the sum of all observations 500?",
        "target_state": "A business dataset contains 50 observations with an average value of 10 units.",
        "target_question": "Is the total across all observations 500 units?",
    },

    {
        "core_id": "C50",
        "label": 0,
        "source_domain": "data_reasoning",
        "target_domain": "business_operations",
        "source_state": "A dataset contains 50 observations. The average value is 8.",
        "source_question": "Is the sum of all observations 500?",
        "target_state": "A business dataset contains 50 observations with an average value of 8 units.",
        "target_question": "Is the total across all observations 500 units?",
    },
]


def build_dataset():
    rows = []

    for case in CASES:

        source_id = f"E6_{case['core_id']}_SRC"
        target_id = f"E6_{case['core_id']}_SHIFT"

        rows.append({
            "case_id": source_id,
            "core_id": case["core_id"],
            "shift_group": "source",
            "source_domain": case["source_domain"],
            "target_domain": case["target_domain"],
            "domain": case["source_domain"],
            "state": case["source_state"],
            "question": case["source_question"],
            "label": case["label"],
        })

        rows.append({
            "case_id": target_id,
            "core_id": case["core_id"],
            "shift_group": "shifted",
            "source_domain": case["source_domain"],
            "target_domain": case["target_domain"],
            "domain": case["target_domain"],
            "state": case["target_state"],
            "question": case["target_question"],
            "label": case["label"],
        })

    return pd.DataFrame(rows)


def validate(df):

    print("=" * 70)
    print("E6 DATASET VALIDATION")
    print("=" * 70)

    assert len(df) == 100
    print("✓ 100 total cases")

    assert df["case_id"].nunique() == 100
    print("✓ Case IDs unique")

    assert df["core_id"].nunique() == 50
    print("✓ 50 matched decision pairs")

    counts = df["label"].value_counts().sort_index().to_dict()

    assert counts.get(0, 0) == 50
    assert counts.get(1, 0) == 50

    print("✓ Labels balanced: 50 false / 50 true")

    shift_counts = df["shift_group"].value_counts().to_dict()

    assert shift_counts["source"] == 50
    assert shift_counts["shifted"] == 50

    print("✓ 50 source / 50 shifted")

    pair_counts = df.groupby("core_id").size()

    assert pair_counts.eq(2).all()

    print("✓ Every core has exactly two cases")

    label_consistency = (
        df.groupby("core_id")["label"]
        .nunique()
    )

    assert label_consistency.eq(1).all()

    print("✓ Labels identical within every matched pair")

    domain_pairs = (
        df.groupby("core_id")
        .agg({
            "source_domain": "nunique",
            "target_domain": "nunique",
        })
    )

    assert domain_pairs["source_domain"].eq(1).all()
    assert domain_pairs["target_domain"].eq(1).all()

    print("✓ Domain mapping consistent")

    print()
    print("SOURCE DOMAINS")
    print(df[df["shift_group"] == "source"]["domain"].value_counts())

    print()
    print("SHIFTED DOMAINS")
    print(df[df["shift_group"] == "shifted"]["domain"].value_counts())

    print()
    print("LABELS BY SHIFT")
    print(
        pd.crosstab(
            df["shift_group"],
            df["label"]
        )
    )

    print()
    print("✓ ALL AUTOMATED VALIDATION CHECKS PASSED")


if __name__ == "__main__":

    df = build_dataset()

    validate(df)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Frozen dataset written to:")
    print(OUTPUT_FILE)
