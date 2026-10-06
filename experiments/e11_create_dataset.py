import os
import pandas as pd
OUTPUT_PATH = "experiments/data/e11_decision_decomposition_dataset.csv"
CASES = [
    # Mathematics

    ("M01", "mathematics", "easy",

     "Seven times eight equals 56.",

     "Seven times eight equals 56.",

     1,

     "56 is divisible by 7.",

     1),



    ("M02", "mathematics", "easy",

     "A square has side length 6 cm.",

     "The area of the square is 36 square centimeters.",

     1,

     "The perimeter of the square is 24 centimeters.",

     1),



    ("M03", "mathematics", "easy",

     "The number 144 is divisible by 12.",

     "144 divided by 12 equals 12.",

     1,

     "144 is divisible by 11.",

     0),



    ("M04", "mathematics", "easy",

     "The average of 10 and 20 is 15.",

     "The sum of 10 and 20 is 30.",

     1,

     "The average of 10 and 20 is 20.",

     0),



    ("M05", "mathematics", "moderate",

     "Three quarters is less than one half.",

     "Three quarters is greater than one half.",

     0,

     "Three quarters is less than one half.",

     0),



    ("M06", "mathematics", "moderate",

     "A rectangle has length 8 cm and width 3 cm.",

     "Its area is 24 square centimeters.",

     1,

     "Its perimeter is 22 centimeters.",

     1),



    ("M07", "mathematics", "moderate",

     "The number 81 is a perfect square.",

     "The square root of 81 is 9.",

     1,

     "81 is the square of 8.",

     0),



    ("M08", "mathematics", "moderate",

     "If x = 5, then 2x + 3 equals 13.",

     "If x = 5, then 2x + 3 equals 13.",

     1,

     "If x = 5, then 2x + 3 equals 14.",

     0),



    ("M09", "mathematics", "hard",

     "A triangle has sides 3, 4, and 5.",

     "The triangle is right-angled.",

     1,

     "The area of the triangle is 6 square units.",

     1),



    ("M10", "mathematics", "hard",

     "The number 36 is divisible by 4.",

     "36 divided by 4 equals 9.",

     1,

     "36 is divisible by 5.",

     0),



    ("M11", "mathematics", "hard",

     "A rectangle has length 12 cm and width 5 cm.",

     "Its area is 60 square centimeters.",

     1,

     "Its area is 70 square centimeters.",

     0),



    ("M12", "mathematics", "hard",

     "The sum of the first five positive integers is 15.",

     "The average of the first five positive integers is 3.",

     1,

     "The sum of the first five positive integers is 20.",

     0),



    # Probability

    ("P01", "probability", "easy",

     "A fair six-sided die is rolled.",

     "The probability of rolling a 3 is 1/6.",

     1,

     "The probability of rolling a 3 is 1/5.",

     0),



    ("P02", "probability", "easy",

     "A fair coin is tossed once.",

     "The probability of heads is 1/2.",

     1,

     "The probability of heads is 1/3.",

     0),



    ("P03", "probability", "easy",

     "A fair six-sided die is rolled.",

     "The probability of rolling a number greater than 6 is 0.",

     1,

     "The probability of rolling a number greater than 6 is 1/6.",

     0),



    ("P04", "probability", "easy",

     "A fair six-sided die is rolled.",

     "The probability of rolling an even number is 1/2.",

     1,

     "The probability of rolling an odd number is 1/2.",

     1),



    ("P05", "probability", "moderate",

     "A bag contains 3 red balls and 2 blue balls.",

     "The probability of drawing a red ball is 3/5.",

     1,

     "The probability of drawing a blue ball is 2/5.",

     1),



    ("P06", "probability", "moderate",

     "A fair six-sided die is rolled.",

     "The probability of rolling a number greater than 3 is 1/2.",

     1,

     "The probability of rolling a number less than 3 is 1/2.",

     0),



    ("P07", "probability", "moderate",

     "A fair coin is tossed.",

     "The probability of tails is 1/2.",

     1,

     "The probability of tails is 1/3.",

     0),



    ("P08", "probability", "moderate",

     "A fair six-sided die is rolled.",

     "The probability of rolling 2 or 5 is 1/3.",

     1,

     "The probability of rolling 2 or 5 is 1/2.",

     0),



    ("P09", "probability", "hard",

     "In a finite sample space, an event has probability 0.",

     "The event is impossible.",

     1,

     "The event must have probability 1.",

     0),



    ("P10", "probability", "hard",

     "A fair six-sided die is rolled.",

     "The probability of rolling 2 or 5 is 1/3.",

     1,

     "The probability of rolling 2 or 5 is 1/2.",

     0),



    ("P11", "probability", "hard",

     "A fair six-sided die is rolled twice.",

     "The probability that both rolls are six is 1/36.",

     1,

     "The probability that both rolls are six is 1/12.",

     0),



    ("P12", "probability", "hard",

     "Two independent fair coin tosses are made.",

     "The probability of getting two heads is 1/4.",

     1,

     "The probability of getting at least one head is 1/4.",

     0),



    # Formal logic

    ("L01", "formal_logic", "easy",

     "All cats are animals. Luna is a cat.",

     "Luna is an animal.",

     1,

     "Luna is a plant.",

     0),



    ("L02", "formal_logic", "easy",

     "All A are B. All B are C.",

     "All A are C.",

     1,

     "No A are C.",

     0),



    ("L03", "formal_logic", "easy",

     "No A are B. X is an A.",

     "X is not a B.",

     1,

     "X is a B.",

     0),



    ("L04", "formal_logic", "easy",

     "All A are B. X is not a B.",

     "X must not be an A.",

     1,

     "X must be an A.",

     0),



    ("L05", "formal_logic", "moderate",

     "If A occurs, then B occurs. A occurs.",

     "B must occur.",

     1,

     "B must not occur.",

     0),



    ("L06", "formal_logic", "moderate",

     "If A occurs, then B occurs. B does not occur.",

     "A cannot have occurred.",

     1,

     "A must have occurred.",

     0),



    ("L07", "formal_logic", "moderate",

     "Some A are B. All B are C.",

     "Some A are C.",

     1,

     "No A are C.",

     0),



    ("L08", "formal_logic", "moderate",

     "No A are B. Some C are A.",

     "Some C are not B.",

     1,

     "All C are B.",

     0),



    ("L09", "formal_logic", "hard",

     "All engineers are professionals. Some engineers are researchers.",

     "Some professionals are researchers.",

     1,

     "No professionals are researchers.",

     0),



    ("L10", "formal_logic", "hard",

     "No mammals are insects. All whales are mammals.",

     "No whales are insects.",

     1,

     "Some whales are insects.",

     0),



    ("L11", "formal_logic", "hard",

     "All birds in a specified group are animals. X is not an animal.",

     "X is not a bird in that group.",

     1,

     "X must be a bird in that group.",

     0),



    ("L12", "formal_logic", "hard",

     "If a device is certified, then it passes inspection. Device X is certified.",

     "Device X passes inspection.",

     1,

     "Device X must fail inspection.",

     0),



    # Science

    ("S01", "science_reasoning", "easy",

     "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",

     "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",

     1,

     "Water freezes at 50 degrees Celsius under standard atmospheric pressure.",

     0),



    ("S02", "science_reasoning", "easy",

     "Earth orbits the Sun.",

     "Earth is a planet orbiting the Sun.",

     1,

     "Earth is a star.",

     0),



    ("S03", "science_reasoning", "easy",

     "Objects near Earth's surface experience gravitational attraction.",

     "Gravity attracts objects toward Earth.",

     1,

     "Gravity repels all masses from Earth.",

     0),



    ("S04", "science_reasoning", "easy",

     "Plants use sunlight during photosynthesis.",

     "Photosynthesis can use light energy.",

     1,

     "Photosynthesis requires no energy input.",

     0),



    ("S05", "science_reasoning", "moderate",

     "Aerobic respiration uses oxygen.",

     "Oxygen can be required for aerobic respiration.",

     1,

     "Oxygen is required for anaerobic respiration.",

     0),



    ("S06", "science_reasoning", "moderate",

     "An object accelerates when its velocity changes.",

     "Acceleration can occur when velocity changes.",

     1,

     "Acceleration requires the object to remain at rest.",

     0),



    ("S07", "science_reasoning", "moderate",

     "Sound requires a medium to propagate.",

     "Sound cannot propagate through a perfect vacuum.",

     1,

     "Sound propagates through a perfect vacuum.",

     0),



    ("S08", "science_reasoning", "moderate",

     "The net work done on an object equals its change in kinetic energy.",

     "Zero net work implies unchanged initial and final kinetic energy.",

     1,

     "Zero net work necessarily means zero motion at every moment.",

     0),



    ("S09", "science_reasoning", "hard",

     "An object has nonzero acceleration.",

     "Its velocity is changing.",

     1,

     "Its speed must be changing at every instant.",

     0),



    ("S10", "science_reasoning", "hard",

     "An object moves in a circle at constant speed.",

     "Its velocity changes direction.",

     1,

     "Its acceleration is necessarily zero.",

     0),



    ("S11", "science_reasoning", "hard",

     "Light can travel through a vacuum.",

     "Light can propagate through empty space.",

     1,

     "Sound can propagate through a perfect vacuum.",

     0),



    ("S12", "science_reasoning", "hard",

     "An isolated object experiences no net external force.",

     "Its acceleration is zero.",

     1,

     "Its velocity must be changing.",

     0),



    # Data reasoning

    ("D01", "data_reasoning", "easy",

     "A dataset contains 2, 4, 6, and 8.",

     "The mean is 5.",

     1,

     "The mean is 6.",

     0),



    ("D02", "data_reasoning", "easy",

     "A dataset contains 1, 3, 5, 7, and 9.",

     "The median is 5.",

     1,

     "The median is 6.",

     0),



    ("D03", "data_reasoning", "easy",

     "A dataset contains 2, 4, 6, and 8.",

     "The range is 6.",

     1,

     "The range is 4.",

     0),



    ("D04", "data_reasoning", "easy",

     "A dataset contains 10, 10, 20, and 20.",

     "The mean is 15.",

     1,

     "The mean is 20.",

     0),



    ("D05", "data_reasoning", "moderate",

     "A dataset has values 2, 4, 6, 8, and 10.",

     "The median is 6.",

     1,

     "The mean is 8.",

     0),



    ("D06", "data_reasoning", "moderate",

     "A dataset has values 1, 2, 3, 4, and 5.",

     "The mean is 3.",

     1,

     "The median is 4.",

     0),



    ("D07", "data_reasoning", "moderate",

     "Every value in a dataset is multiplied by 3.",

     "The median is also multiplied by 3.",

     1,

     "The median remains unchanged.",

     0),



    ("D08", "data_reasoning", "moderate",

     "A dataset contains 5, 5, 5, 10, and 20.",

     "The mode is 5.",

     1,

     "The mode is 10.",

     0),



    ("D09", "data_reasoning", "hard",

     "A dataset has values 2, 4, 6, 8, and 10.",

     "The mean and median are both 6.",

     1,

     "The mean is 8 and the median is 6.",

     0),



    ("D10", "data_reasoning", "hard",

     "A dataset has values 1, 2, 3, 100.",

     "The mean is 26.5.",

     1,

     "The median is 26.5.",

     0),



    ("D11", "data_reasoning", "hard",

     "A dataset has values 2, 4, 6, 8, and 10.",

     "Removing the minimum value changes the mean.",

     1,

     "Removing the minimum value leaves the mean unchanged.",

     0),



    ("D12", "data_reasoning", "hard",

     "A dataset has values 3, 3, 5, 7, and 9.",

     "The median is 5.",

     1,

     "The median is 7.",

     0),

]



# -------------------------------------------------------------------

# Balance correction:

# We need exactly 30 compound-TRUE cases.

# A compound decision is TRUE iff both A and B are TRUE.

# The original construction produced only 6 such cases.

# These 24 B propositions are replaced with objectively TRUE

# propositions while preserving the A proposition for each case.

# -------------------------------------------------------------------



BALANCE_PATCHES = {

    "M03": "Twelve multiplied by 12 equals 144.",

    "M04": "Thirty divided by 2 equals 15.",

    "M07": "Nine multiplied by 9 equals 81.",

    "M08": "Twice 5 plus 3 equals 13.",

    "M10": "Four multiplied by 9 equals 36.",

    "M11": "The area of the rectangle is 60 square centimeters.",

    "M12": "Fifteen divided equally among five values gives 3 per value.",



    "P01": "The probability of rolling a 3 on a fair six-sided die is 1/6.",

    "P02": "The probability of getting tails from a fair coin is 1/2.",

    "P03": "The probability of rolling a number less than 7 on a fair six-sided die is 1.",

    "P06": "The probability of rolling a number greater than 3 on a fair six-sided die is 1/2.",

    "P07": "The probability of getting heads from a fair coin is 1/2.",

    "P08": "The probability of rolling 2 or 5 on a fair six-sided die is 1/3.",

    "P09": "In a finite sample space, an event with probability 0 is impossible.",

    "P10": "The probability of rolling 2 or 5 on a fair six-sided die is 1/3.",



    "L01": "Luna belongs to the category of animals.",

    "L02": "Anything that is an A is also a C.",

    "L03": "No member of category A is a member of category B, so X cannot be a B.",

    "L04": "X cannot belong to category A.",



    "S01": "The freezing point of water is 0 degrees Celsius at standard atmospheric pressure.",

    "S02": "The Sun is orbited by Earth.",

    "S03": "Earth exerts gravitational attraction on nearby objects.",

    "S04": "Light energy can be used during photosynthesis.",



    "D01": "The mean of 2, 4, 6, and 8 is 5."

}

def main():
    rows = []

    for (
        case_id,
        domain,
        difficulty,
        a_state,
        a_question,
        a_label,
        b_question,
        b_label,
    ) in CASES:

        if case_id in BALANCE_PATCHES:
            b_question = BALANCE_PATCHES[case_id]
            b_label = 1

        compound_label = int(a_label == 1 and b_label == 1)

        rows.append(
            {
                "case_id": case_id,
                "domain": domain,
                "difficulty": difficulty,
                "label": compound_label,
                "A_state": a_state,
                "A_question": a_question,
                "A_label": a_label,
                "B_state": a_state,
                "B_question": b_question,
                "B_label": b_label,
                "compound_state": a_state,
                "compound_question": (
                    "Are both of the following statements true? "
                    f"Statement A: {a_question} "
                    f"Statement B: {b_question}"
                ),
                "distractor_1_question": (
                    "A triangle has three interior angles whose measures sum to 180 degrees. "
                    "Is the sum of its interior angles 180 degrees?"
                ),
                "distractor_2_question": (
                    "The number 2 is added to itself. Is the resulting value 4?"
                ),
            }
        )

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("E11 DECISION DECOMPOSITION DATASET")
    print("=" * 70)
    print("Total cases:", len(df))

    print("Labels:")
    print(df["label"].value_counts().sort_index())

    print("\nDomains:")
    print(df["domain"].value_counts().sort_index())

    print("\nDifficulty:")
    print(df["difficulty"].value_counts().sort_index())

    print("\nSaved to:", OUTPUT_PATH)

    assert len(df) == 60
    assert df["case_id"].nunique() == 60
    assert df["label"].value_counts().to_dict() == {0: 30, 1: 30}
    assert df["domain"].value_counts().eq(12).all()
    assert df["difficulty"].value_counts().eq(20).all()
    assert (
        df["label"]
        == ((df["A_label"] == 1) & (df["B_label"] == 1)).astype(int)
    ).all()

    print("\nValidation: PASSED")


if __name__ == "__main__":
    main()
