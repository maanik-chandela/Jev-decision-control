"""
E5: Adversarial Robustness Dataset Construction

Purpose:
    Construct a frozen, objectively answerable dataset for evaluating
    JEV robustness to controlled adversarial transformations.

Design:
    60 original cases
    30 true / 30 false
    5 domains x 12 cases

    Each original receives 2 adversarial variants:
        - polarity / negation
        - distractor insertion
        - structural complexity

Important:
    No JEV outputs are used during dataset construction.
    Ground-truth labels are assigned before API evaluation.
"""

from pathlib import Path
import csv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "experiments" / "data"

OUTPUT_FILE = DATA_DIR / "e5_adversarial_dataset.csv"


# ============================================================
# DATASET
# ============================================================

CASES = [

    # ========================================================
    # MATHEMATICS — 12
    # ========================================================

    {
        "id": "E5_M01",
        "domain": "mathematics",
        "state": "A rectangle has length 8 cm and width 5 cm.",
        "question": "Is its area 40 square centimeters?",
        "label": 1,
    },
    {
        "id": "E5_M02",
        "domain": "mathematics",
        "state": "A number x satisfies x + 7 = 12.",
        "question": "Is x equal to 5?",
        "label": 1,
    },
    {
        "id": "E5_M03",
        "domain": "mathematics",
        "state": "A triangle has base 10 cm and height 6 cm.",
        "question": "Is its area 30 square centimeters?",
        "label": 1,
    },
    {
        "id": "E5_M04",
        "domain": "mathematics",
        "state": "A circle has radius 3 cm.",
        "question": "Is its diameter 6 cm?",
        "label": 1,
    },
    {
        "id": "E5_M05",
        "domain": "mathematics",
        "state": "The average of 4, 8, and 12 is calculated.",
        "question": "Is the average 8?",
        "label": 1,
    },
    {
        "id": "E5_M06",
        "domain": "mathematics",
        "state": "A square has side length 7 cm.",
        "question": "Is its perimeter 28 cm?",
        "label": 1,
    },

    {
        "id": "E5_M07",
        "domain": "mathematics",
        "state": "A rectangle has length 9 cm and width 4 cm.",
        "question": "Is its area 45 square centimeters?",
        "label": 0,
    },
    {
        "id": "E5_M08",
        "domain": "mathematics",
        "state": "A number x satisfies 2x = 18.",
        "question": "Is x equal to 8?",
        "label": 0,
    },
    {
        "id": "E5_M09",
        "domain": "mathematics",
        "state": "A triangle has base 12 cm and height 5 cm.",
        "question": "Is its area 60 square centimeters?",
        "label": 0,
    },
    {
        "id": "E5_M10",
        "domain": "mathematics",
        "state": "A circle has radius 4 cm.",
        "question": "Is its diameter 10 cm?",
        "label": 0,
    },
    {
        "id": "E5_M11",
        "domain": "mathematics",
        "state": "The average of 6, 10, and 14 is calculated.",
        "question": "Is the average 12?",
        "label": 0,
    },
    {
        "id": "E5_M12",
        "domain": "mathematics",
        "state": "A square has side length 6 cm.",
        "question": "Is its perimeter 24 square centimeters?",
        "label": 0,
    },


    # ========================================================
    # PROBABILITY — 12
    # ========================================================

    {
        "id": "E5_P01",
        "domain": "probability",
        "state": "A fair six-sided die is rolled once.",
        "question": "Is the probability of rolling a 6 equal to 1/6?",
        "label": 1,
    },
    {
        "id": "E5_P02",
        "domain": "probability",
        "state": "A fair coin is tossed twice.",
        "question": "Is the probability of getting two heads equal to 1/4?",
        "label": 1,
    },
    {
        "id": "E5_P03",
        "domain": "probability",
        "state": "A bag contains 3 red balls and 7 blue balls.",
        "question": "Is the probability of drawing a red ball on one draw equal to 0.3?",
        "label": 1,
    },
    {
        "id": "E5_P04",
        "domain": "probability",
        "state": "A fair six-sided die is rolled once.",
        "question": "Is the probability of rolling an even number equal to 1/2?",
        "label": 1,
    },
    {
        "id": "E5_P05",
        "domain": "probability",
        "state": "Two independent fair coin tosses are performed.",
        "question": "Is the probability that at least one toss is heads equal to 3/4?",
        "label": 1,
    },
    {
        "id": "E5_P06",
        "domain": "probability",
        "state": "A standard deck contains 52 cards, including 13 hearts.",
        "question": "Is the probability of drawing a heart on one random draw equal to 1/4?",
        "label": 1,
    },

    {
        "id": "E5_P07",
        "domain": "probability",
        "state": "A fair six-sided die is rolled once.",
        "question": "Is the probability of rolling a 6 equal to 1/3?",
        "label": 0,
    },
    {
        "id": "E5_P08",
        "domain": "probability",
        "state": "A fair coin is tossed twice.",
        "question": "Is the probability of getting two heads equal to 1/2?",
        "label": 0,
    },
    {
        "id": "E5_P09",
        "domain": "probability",
        "state": "A bag contains 2 red balls and 8 blue balls.",
        "question": "Is the probability of drawing a red ball equal to 0.5?",
        "label": 0,
    },
    {
        "id": "E5_P10",
        "domain": "probability",
        "state": "A fair six-sided die is rolled once.",
        "question": "Is the probability of rolling an odd number equal to 1/3?",
        "label": 0,
    },
    {
        "id": "E5_P11",
        "domain": "probability",
        "state": "Two independent fair coin tosses are performed.",
        "question": "Is the probability that at least one toss is heads equal to 1/2?",
        "label": 0,
    },
    {
        "id": "E5_P12",
        "domain": "probability",
        "state": "A standard deck contains 52 cards, including 13 hearts.",
        "question": "Is the probability of drawing a heart equal to 1/2?",
        "label": 0,
    },


    # ========================================================
    # FORMAL LOGIC — 12
    # ========================================================

    {
        "id": "E5_L01",
        "domain": "formal_logic",
        "state": "Every A is a B. Every B is a C.",
        "question": "Must every A be a C?",
        "label": 1,
    },
    {
        "id": "E5_L02",
        "domain": "formal_logic",
        "state": "Every cat is a mammal. Luna is a cat.",
        "question": "Must Luna be a mammal?",
        "label": 1,
    },
    {
        "id": "E5_L03",
        "domain": "formal_logic",
        "state": "No mammals are reptiles. A tiger is a mammal.",
        "question": "Must the tiger be a non-reptile?",
        "label": 1,
    },
    {
        "id": "E5_L04",
        "domain": "formal_logic",
        "state": "If P is true, then Q is true. P is true.",
        "question": "Must Q be true?",
        "label": 1,
    },
    {
        "id": "E5_L05",
        "domain": "formal_logic",
        "state": "Some students are athletes. Every athlete trains regularly.",
        "question": "Must at least one student train regularly?",
        "label": 1,
    },
    {
        "id": "E5_L06",
        "domain": "formal_logic",
        "state": "Every engineer studied mathematics. Ravi is an engineer.",
        "question": "Must Ravi have studied mathematics?",
        "label": 1,
    },

    {
        "id": "E5_L07",
        "domain": "formal_logic",
        "state": "Every A is a B. Some C is a B.",
        "question": "Must every C be an A?",
        "label": 0,
    },
    {
        "id": "E5_L08",
        "domain": "formal_logic",
        "state": "If P is true, then Q is true. Q is true.",
        "question": "Must P be true?",
        "label": 0,
    },
    {
        "id": "E5_L09",
        "domain": "formal_logic",
        "state": "All doctors are educated. Some educated people are musicians.",
        "question": "Must every musician be a doctor?",
        "label": 0,
    },
    {
        "id": "E5_L10",
        "domain": "formal_logic",
        "state": "If A occurs, then B occurs. B occurs.",
        "question": "Must A have occurred?",
        "label": 0,
    },
    {
        "id": "E5_L11",
        "domain": "formal_logic",
        "state": "Every bird can fly. Penguins are birds.",
        "question": "Does the stated premise logically imply that penguins cannot fly?",
        "label": 0,
    },
    {
        "id": "E5_L12",
        "domain": "formal_logic",
        "state": "Some A are B. No B are C.",
        "question": "Must every A be non-C?",
        "label": 0,
    },


    # ========================================================
    # SCIENCE REASONING — 12
    # ========================================================

    {
        "id": "E5_S01",
        "domain": "science_reasoning",
        "state": "An object travels 100 meters in 10 seconds at constant speed.",
        "question": "Is its speed 10 meters per second?",
        "label": 1,
    },
    {
        "id": "E5_S02",
        "domain": "science_reasoning",
        "state": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "question": "Does water freeze at 0 degrees Celsius under these conditions?",
        "label": 1,
    },
    {
        "id": "E5_S03",
        "domain": "science_reasoning",
        "state": "An object has mass 2 kg and acceleration 3 m/s².",
        "question": "Is the net force 6 newtons?",
        "label": 1,
    },
    {
        "id": "E5_S04",
        "domain": "science_reasoning",
        "state": "At constant pressure, an ideal gas is heated from 300 K to 600 K.",
        "question": "Does its volume double?",
        "label": 1,
    },
    {
        "id": "E5_S05",
        "domain": "science_reasoning",
        "state": "A metal rod is heated and expands.",
        "question": "Does its length increase as a result of thermal expansion?",
        "label": 1,
    },
    {
        "id": "E5_S06",
        "domain": "science_reasoning",
        "state": "A circuit has voltage 12 V and resistance 4 ohms.",
        "question": "Is the current 3 amperes?",
        "label": 1,
    },

    {
        "id": "E5_S07",
        "domain": "science_reasoning",
        "state": "An object travels 120 meters in 10 seconds at constant speed.",
        "question": "Is its speed 15 meters per second?",
        "label": 0,
    },
    {
        "id": "E5_S08",
        "domain": "science_reasoning",
        "state": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "question": "Does water freeze at 100 degrees Celsius under these conditions?",
        "label": 0,
    },
    {
        "id": "E5_S09",
        "domain": "science_reasoning",
        "state": "An object has mass 5 kg and acceleration 2 m/s².",
        "question": "Is the net force 20 newtons?",
        "label": 0,
    },
    {
        "id": "E5_S10",
        "domain": "science_reasoning",
        "state": "At constant pressure, an ideal gas is heated from 300 K to 600 K.",
        "question": "Does its volume become half its original value?",
        "label": 0,
    },
    {
        "id": "E5_S11",
        "domain": "science_reasoning",
        "state": "A circuit has voltage 12 V and resistance 6 ohms.",
        "question": "Is the current 3 amperes?",
        "label": 0,
    },
    {
        "id": "E5_S12",
        "domain": "science_reasoning",
        "state": "An object is moving at constant velocity in a straight line.",
        "question": "Must a nonzero net force be acting on it?",
        "label": 0,
    },


    # ========================================================
    # DATA / QUANTITATIVE REASONING — 12
    # ========================================================

    {
        "id": "E5_D01",
        "domain": "data_reasoning",
        "state": "A classifier correctly predicts 95 of 100 examples.",
        "question": "Is its accuracy 95%?",
        "label": 1,
    },
    {
        "id": "E5_D02",
        "domain": "data_reasoning",
        "state": "A classifier has TP=80 and FN=20.",
        "question": "Is its recall 80%?",
        "label": 1,
    },
    {
        "id": "E5_D03",
        "domain": "data_reasoning",
        "state": "A confusion matrix has TP=80 and FP=20.",
        "question": "Is its precision 80%?",
        "label": 1,
    },
    {
        "id": "E5_D04",
        "domain": "data_reasoning",
        "state": "A dataset contains 50 positive and 50 negative examples.",
        "question": "Are the two classes equally represented?",
        "label": 1,
    },
    {
        "id": "E5_D05",
        "domain": "data_reasoning",
        "state": "Five values are 2, 4, 6, 8, and 10.",
        "question": "Is their median 6?",
        "label": 1,
    },
    {
        "id": "E5_D06",
        "domain": "data_reasoning",
        "state": "A quantity increases from 50 to 60.",
        "question": "Is the percentage increase 20%?",
        "label": 1,
    },

    {
        "id": "E5_D07",
        "domain": "data_reasoning",
        "state": "A classifier correctly predicts 80 of 100 examples.",
        "question": "Is its accuracy 90%?",
        "label": 0,
    },
    {
        "id": "E5_D08",
        "domain": "data_reasoning",
        "state": "A classifier has TP=60 and FN=40.",
        "question": "Is its recall 80%?",
        "label": 0,
    },
    {
        "id": "E5_D09",
        "domain": "data_reasoning",
        "state": "A confusion matrix has TP=50 and FP=50.",
        "question": "Is its precision 80%?",
        "label": 0,
    },
    {
        "id": "E5_D10",
        "domain": "data_reasoning",
        "state": "A dataset contains 80 positive and 20 negative examples.",
        "question": "Are the two classes equally represented?",
        "label": 0,
    },
    {
        "id": "E5_D11",
        "domain": "data_reasoning",
        "state": "Five values are 2, 4, 8, 10, and 12.",
        "question": "Is their median 7?",
        "label": 0,
    },
    {
        "id": "E5_D12",
        "domain": "data_reasoning",
        "state": "A quantity increases from 80 to 100.",
        "question": "Is the percentage increase 20%?",
        "label": 0,
    },
]


# ============================================================
# ADVERSARIAL TRANSFORMATIONS
# ============================================================

def make_negation(case):
    """
    Controlled polarity transformation.

    The transformed question asks the logically equivalent
    proposition using explicit negation.
    """

    question = case["question"]

    replacements = {
        "Is its area 40 square centimeters?":
            "Is it false that its area is not 40 square centimeters?",

        "Is x equal to 5?":
            "Is it false that x is not equal to 5?",

        "Is its area 30 square centimeters?":
            "Is it false that its area is not 30 square centimeters?",

        "Is its diameter 6 cm?":
            "Is it false that its diameter is not 6 cm?",

        "Is the average 8?":
            "Is it false that the average is not 8?",

        "Is its perimeter 28 cm?":
            "Is it false that its perimeter is not 28 cm?",

        "Is its area 45 square centimeters?":
            "Is it false that its area is not 45 square centimeters?",

        "Is x equal to 8?":
            "Is it false that x is not equal to 8?",

        "Is its area 60 square centimeters?":
            "Is it false that its area is not 60 square centimeters?",

        "Is its diameter 10 cm?":
            "Is it false that its diameter is not 10 cm?",

        "Is the average 12?":
            "Is it false that the average is not 12?",

        "Is its perimeter 24 square centimeters?":
            "Is it false that its perimeter is not 24 square centimeters?",

        "Is the probability of rolling a 6 equal to 1/6?":
            "Is it false that the probability of rolling a 6 is not 1/6?",

        "Is the probability of getting two heads equal to 1/4?":
            "Is it false that the probability of getting two heads is not 1/4?",

        "Is the probability of drawing a red ball on one draw equal to 0.3?":
            "Is it false that the probability of drawing a red ball is not 0.3?",

        "Is the probability of rolling an even number equal to 1/2?":
            "Is it false that the probability of rolling an even number is not 1/2?",

        "Is the probability that at least one toss is heads equal to 3/4?":
            "Is it false that the probability of at least one heads is not 3/4?",

        "Is the probability of drawing a heart on one random draw equal to 1/4?":
            "Is it false that the probability of drawing a heart is not 1/4?",

        "Is the probability of rolling a 6 equal to 1/3?":
            "Is it false that the probability of rolling a 6 is not 1/3?",

        "Is the probability of getting two heads equal to 1/2?":
            "Is it false that the probability of getting two heads is not 1/2?",

        "Is the probability of drawing a red ball equal to 0.5?":
            "Is it false that the probability of drawing a red ball is not 0.5?",

        "Is the probability of rolling an odd number equal to 1/3?":
            "Is it false that the probability of rolling an odd number is not 1/3?",

        "Is the probability that at least one toss is heads equal to 1/2?":
            "Is it false that the probability of at least one heads is not 1/2?",

        "Is the probability of drawing a heart equal to 1/2?":
            "Is it false that the probability of drawing a heart is not 1/2?",
    }

    if question in replacements:
        return replacements[question]

    return f"Is it false that the following statement is false: {question}"


def make_distractor(case):
    """
    Add irrelevant information while preserving the original
    state and proposition.
    """

    distractors = {
        "mathematics":
            "The calculation is part of a classroom exercise. "
            "The student wrote the result in a notebook before lunch. ",

        "probability":
            "The experiment is being performed as part of a statistics lesson. "
            "The equipment was prepared earlier in the day. ",

        "formal_logic":
            "The statements are being considered in a formal reasoning exercise. "
            "No additional premises are provided beyond those stated. ",

        "science_reasoning":
            "The measurement is part of a basic science exercise. "
            "The observation was recorded in a laboratory notebook. ",

        "data_reasoning":
            "The calculation is being used in a data-analysis exercise. "
            "The values were recorded in a table before the calculation. ",
    }

    return (
        distractors[case["domain"]]
        + case["state"]
    )


def make_structural(case):
    """
    Increase syntactic/logical complexity while preserving
    the proposition.
    """

    transformations = {
        "E5_M01":
            "Given that the rectangle's length is 8 cm and its width is 5 cm, "
            "is the resulting area necessarily 40 square centimeters?",

        "E5_M02":
            "If x is a number for which adding 7 results in 12, "
            "does it follow that x must be 5?",

        "E5_M03":
            "Given a triangular region whose base is 10 cm and height is 6 cm, "
            "does the standard area formula imply an area of 30 square centimeters?",

        "E5_M04":
            "For a circle whose radius is specified as 3 cm, "
            "is a diameter of 6 cm necessarily implied?",

        "E5_M05":
            "When the three values 4, 8, and 12 are averaged arithmetically, "
            "is the resulting mean equal to 8?",

        "E5_M06":
            "For a square with each side measuring 7 cm, "
            "does the definition of perimeter imply a total boundary length of 28 cm?",

        "E5_M07":
            "Given that a rectangle measures 9 cm by 4 cm, "
            "does its area necessarily equal 45 square centimeters?",

        "E5_M08":
            "If twice a number x is equal to 18, "
            "does that condition necessarily make x equal to 8?",

        "E5_M09":
            "For a triangle with a 12 cm base and a 5 cm height, "
            "does the standard area calculation give 60 square centimeters?",

        "E5_M10":
            "If the radius of a circle is 4 cm, "
            "does this information necessarily imply that its diameter is 10 cm?",

        "E5_M11":
            "When 6, 10, and 14 are combined using the arithmetic mean, "
            "is the resulting average equal to 12?",

        "E5_M12":
            "For a square whose side length is 6 cm, "
            "does the perimeter formula imply a value of 24 square centimeters?",

        "E5_P01":
            "Given one independent roll of a fair six-sided die, "
            "is the probability assigned to the outcome of obtaining a 6 equal to 1/6?",

        "E5_P02":
            "For two independent tosses of a fair coin, "
            "does the joint event that both tosses are heads have probability 1/4?",

        "E5_P03":
            "With 3 red and 7 blue balls in the bag, "
            "does a uniformly random single draw assign probability 0.3 to the red outcome?",

        "E5_P04":
            "Under a single roll of a fair six-sided die, "
            "does the event that the outcome is even have probability 1/2?",

        "E5_P05":
            "For two independent fair coin tosses, "
            "is the complement-based probability of observing at least one head equal to 3/4?",

        "E5_P06":
            "Given 13 hearts among the 52 cards in a standard deck, "
            "does a uniformly random single-card draw have probability 1/4 of producing a heart?",

        "E5_P07":
            "For one fair six-sided die roll, "
            "does the event of obtaining a 6 necessarily have probability 1/3?",

        "E5_P08":
            "When a fair coin is independently tossed twice, "
            "is the probability of the joint heads-heads outcome 1/2?",

        "E5_P09":
            "Given a bag containing 2 red and 8 blue balls, "
            "does one uniformly random draw assign probability 0.5 to drawing red?",

        "E5_P10":
            "For one roll of a fair six-sided die, "
            "does the collection of odd outcomes have probability 1/3?",

        "E5_P11":
            "For two independent fair coin tosses, "
            "is the probability of the event consisting of at least one head equal to 1/2?",

        "E5_P12":
            "Considering the 13 hearts among 52 cards in the standard deck, "
            "does a uniformly random draw produce a heart with probability 1/2?",

        "E5_L01":
            "Given that membership in A necessarily implies membership in B, "
            "and membership in B necessarily implies membership in C, "
            "is membership in C therefore necessarily implied for every member of A?",

        "E5_L02":
            "If every cat belongs to the mammal category and Luna is known to be a cat, "
            "does the stated information necessarily entail that Luna is a mammal?",

        "E5_L03":
            "Given the premise that no mammal is a reptile and that the tiger is a mammal, "
            "is the tiger necessarily excluded from the reptile category?",

        "E5_L04":
            "If P being true is sufficient for Q to be true, and P is in fact true, "
            "must Q consequently be true?",

        "E5_L05":
            "If at least one student belongs to the athlete group and every athlete trains regularly, "
            "does it necessarily follow that at least one student trains regularly?",

        "E5_L06":
            "If every engineer has studied mathematics and Ravi is an engineer, "
            "does the universal premise necessarily imply that Ravi studied mathematics?",

        "E5_L07":
            "Given only that every A is a B and that at least one C is a B, "
            "is it logically necessary that every C is an A?",

        "E5_L08":
            "If P is sufficient for Q but Q is merely known to be true, "
            "does the information logically force P to be true?",

        "E5_L09":
            "Given that all doctors are educated and that some educated people are musicians, "
            "does this logically require every musician to be a doctor?",

        "E5_L10":
            "If A occurs, then B occurs, and B is observed to occur, "
            "does the stated information necessarily imply that A occurred?",
        "E5_L11":
            "Given that every bird can fly and penguins are birds, "
            "does the stated information logically imply that penguins cannot fly?",
        "E5_L12":
            "Given that some A are B and no B are C, "
            "must every A be non-C?",
        "E5_S01":
            "Given 100 meters of displacement covered in 10 seconds at constant speed, "
            "does the standard speed calculation produce 10 meters per second?",

        "E5_S02":
            "Under standard atmospheric pressure, given that water freezes at 0 degrees Celsius, "
            "does the stated condition imply freezing at 0 degrees Celsius?",

        "E5_S03":
            "For an object of mass 2 kg undergoing acceleration of 3 m/s², "
            "does Newton's second law give a net force of 6 newtons?",

        "E5_S04":
            "For an ideal gas heated from 300 K to 600 K while pressure remains constant, "
            "does the proportional temperature-volume relationship imply a doubling of volume?",

        "E5_S05":
            "Given that a metal rod is heated and undergoes thermal expansion, "
            "does this process imply an increase in its length?",

        "E5_S06":
            "With a potential difference of 12 V across a resistance of 4 ohms, "
            "does Ohm's law imply a current of 3 amperes?",

        "E5_S07":
            "For an object traveling 120 meters in 10 seconds at constant speed, "
            "does the resulting speed necessarily equal 15 meters per second?",

        "E5_S08":
            "Under standard atmospheric pressure, does the stated freezing condition for water "
            "imply that water freezes at 100 degrees Celsius?",

        "E5_S09":
            "Given mass 5 kg and acceleration 2 m/s², "
            "does Newton's second law imply a net force of 20 newtons?",

        "E5_S10":
            "For an ideal gas heated from 300 K to 600 K at constant pressure, "
            "does the temperature-volume relationship require its volume to become half its initial value?",

        "E5_S11":
            "Given 12 V across a resistance of 6 ohms, "
            "does Ohm's law require the current to be 3 amperes?",

        "E5_S12":
            "If an object moves at constant velocity along a straight line, "
            "does this condition necessarily require a nonzero net force?",

        "E5_D01":
            "Given that 95 out of 100 classification decisions are correct, "
            "does the corresponding accuracy equal 95%?",

        "E5_D02":
            "With 80 true positives and 20 false negatives, "
            "does the recall calculation necessarily produce 80%?",

        "E5_D03":
            "For a classifier having 80 true positives and 20 false positives, "
            "does precision equal 80%?",

        "E5_D04":
            "If a dataset contains exactly 50 positive and 50 negative examples, "
            "does that composition imply equal representation of the two classes?",

        "E5_D05":
            "For the ordered values 2, 4, 6, 8, and 10, "
            "is the central value, and therefore the median, equal to 6?",

        "E5_D06":
            "When a quantity changes from 50 to 60, "
            "does calculating the relative increase with respect to the original value give 20%?",

        "E5_D07":
            "If 80 out of 100 predictions are correct, "
            "does the corresponding accuracy necessarily equal 90%?",

        "E5_D08":
            "Given 60 true positives and 40 false negatives, "
            "does the recall formula produce 80%?",

        "E5_D09":
            "For 50 true positives and 50 false positives, "
            "does the resulting precision equal 80%?",

        "E5_D10":
            "If 80 examples are positive and 20 are negative, "
            "does the dataset necessarily contain equally represented classes?",

        "E5_D11":
            "For the values 2, 4, 8, 10, and 12, "
            "is their median 7?",
        "E5_D12":
            "When a quantity rises from 80 to 100, "
            "does the relative increase measured against the original value equal 20%?",
    }

    return transformations[case["id"]]


# ============================================================
# BUILD VARIANTS
# ============================================================

rows = []

for case in CASES:

    # Variant 1: polarity / negation
    rows.append({
        "counterfactual_id": f"{case['id']}_V1",
        "original_id": case["id"],
        "domain": case["domain"],
        "perturbation_type": "negation",
        "state": case["state"],
        "question": make_negation(case),
        "label": case["label"],
    })

    # Variant 2: irrelevant distractor information
    rows.append({
        "counterfactual_id": f"{case['id']}_V2",
        "original_id": case["id"],
        "domain": case["domain"],
        "perturbation_type": "distractor",
        "state": make_distractor(case),
        "question": case["question"],
        "label": case["label"],
    })

    # Variant 3: increased structural complexity
    rows.append({
        "counterfactual_id": f"{case['id']}_V3",
        "original_id": case["id"],
        "domain": case["domain"],
        "perturbation_type": "structural_complexity",
        "state": case["state"],
        "question": make_structural(case),
        "label": case["label"],
    })

# ============================================================
# VALIDATION
# ============================================================

# ---------------------------------------------------------------------------

print("=" * 70)
print("E5 DATASET VALIDATION")
print("=" * 70)

assert len(CASES) == 60
assert len(rows) == 180

# Original class balance
true_originals = sum(c["label"] == 1 for c in CASES)
false_originals = sum(c["label"] == 0 for c in CASES)

assert true_originals == 30
assert false_originals == 30

# Domain balance
domains = {}

for case in CASES:
    domains.setdefault(case["domain"], 0)
    domains[case["domain"]] += 1

assert len(domains) == 5
assert all(count == 12 for count in domains.values())

# Every original has exactly 3 variants
variant_counts = {}

for row in rows:
    variant_counts.setdefault(row["original_id"], 0)
    variant_counts[row["original_id"]] += 1

assert all(count == 3 for count in variant_counts.values())

# Perturbation balance
perturbation_counts = {}

for row in rows:
    perturbation_counts.setdefault(row["perturbation_type"], 0)
    perturbation_counts[row["perturbation_type"]] += 1

assert perturbation_counts["negation"] == 60
assert perturbation_counts["distractor"] == 60
assert perturbation_counts["structural_complexity"] == 60

# Unique IDs
ids = [row["counterfactual_id"] for row in rows]
assert len(ids) == len(set(ids))

# No missing fields
required_fields = [
    "counterfactual_id",
    "original_id",
    "domain",
    "perturbation_type",
    "state",
    "question",
    "label",
]

for row in rows:
    for field in required_fields:
        assert row[field] not in ("", None)

# Ground truth inheritance
original_labels = {
    case["id"]: case["label"]
    for case in CASES
}

for row in rows:
    assert row["label"] == original_labels[row["original_id"]]


# ============================================================
# WRITE CSV
# ============================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "counterfactual_id",
    "original_id",
    "domain",
    "perturbation_type",
    "state",
    "question",
    "label",
]

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

print()
print("VALIDATION PASSED")
print()
print(f"Original cases: {len(CASES)}")
print(f"Counterfactual cases: {len(rows)}")
print()
print("Original labels:")
print(f"  True:  {true_originals}")
print(f"  False: {false_originals}")
print()
print("Domains:")

for domain, count in sorted(domains.items()):
    print(f"  {domain}: {count}")

print()
print("Perturbation types:")

for perturbation, count in sorted(perturbation_counts.items()):
    print(f"  {perturbation}: {count}")

print()
print(f"Output: {OUTPUT_FILE}")
print()
print("IMPORTANT:")
print("The dataset is now frozen.")
print("Do not modify cases based on JEV outputs.")
print("Run JEV only after reviewing this dataset.")