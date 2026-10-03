import csv
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "experiments" / "data"

OUTPUT = DATA / "e4b_hard_candidates.csv"


cases = [

    # ---------------------------------------------------------
    # MATHEMATICS
    # ---------------------------------------------------------

    {
        "candidate_id": "H001",
        "domain": "mathematics",
        "state": "A number x satisfies 2x + 3 = 11.",
        "question": "Is x greater than 5?",
        "true_label": 0,
    },
    {
        "candidate_id": "H002",
        "domain": "mathematics",
        "state": "A number x satisfies 3x - 4 = 11.",
        "question": "Is x greater than 4?",
        "true_label": 1,
    },
    {
        "candidate_id": "H003",
        "domain": "mathematics",
        "state": "A rectangle has length 8 cm and width 5 cm.",
        "question": "Is its perimeter greater than 20 cm?",
        "true_label": 1,
    },
    {
        "candidate_id": "H004",
        "domain": "mathematics",
        "state": "A rectangle has length 7 cm and width 4 cm.",
        "question": "Is its area greater than 30 square centimeters?",
        "true_label": 0,
    },
    {
        "candidate_id": "H005",
        "domain": "mathematics",
        "state": "A bag contains 3 red balls and 7 blue balls. One ball is selected uniformly at random.",
        "question": "Is the probability of selecting a red ball greater than 0.25?",
        "true_label": 1,
    },
    {
        "candidate_id": "H006",
        "domain": "mathematics",
        "state": "A bag contains 4 red balls and 6 blue balls. One ball is selected uniformly at random.",
        "question": "Is the probability of selecting a red ball exactly 0.5?",
        "true_label": 0,
    },
    {
        "candidate_id": "H007",
        "domain": "mathematics",
        "state": "A fair six-sided die is rolled once.",
        "question": "Is the probability of obtaining an even number greater than 0.4?",
        "true_label": 1,
    },
    {
        "candidate_id": "H008",
        "domain": "mathematics",
        "state": "A fair six-sided die is rolled once.",
        "question": "Is the probability of obtaining a number greater than 4 less than 0.4?",
        "true_label": 1,
    },
    {
        "candidate_id": "H009",
        "domain": "mathematics",
        "state": "The mean of five numbers is 12.",
        "question": "Is the sum of the five numbers exactly 60?",
        "true_label": 1,
    },
    {
        "candidate_id": "H010",
        "domain": "mathematics",
        "state": "The mean of four numbers is 15.",
        "question": "Is the sum of the four numbers greater than 50?",
        "true_label": 1,
    },
    {
        "candidate_id": "H011",
        "domain": "mathematics",
        "state": "A price is increased by 20% and then decreased by 20%.",
        "question": "Is the final price equal to the original price?",
        "true_label": 0,
    },
    {
        "candidate_id": "H012",
        "domain": "mathematics",
        "state": "A price is decreased by 25% and then increased by 25%.",
        "question": "Is the final price greater than the original price?",
        "true_label": 0,
    },
    {
        "candidate_id": "H013",
        "domain": "mathematics",
        "state": "A triangle has side lengths 3 cm, 4 cm, and 5 cm.",
        "question": "Is the triangle right-angled?",
        "true_label": 1,
    },
    {
        "candidate_id": "H014",
        "domain": "mathematics",
        "state": "A triangle has side lengths 2 cm, 3 cm, and 6 cm.",
        "question": "Can these three lengths form a triangle?",
        "true_label": 0,
    },
    {
        "candidate_id": "H015",
        "domain": "mathematics",
        "state": "The sequence is 2, 4, 8, 16, 32.",
        "question": "Is the next number necessarily 64?",
        "true_label": 1,
    },

    # ---------------------------------------------------------
    # LOGICAL REASONING
    # ---------------------------------------------------------

    {
        "candidate_id": "H016",
        "domain": "reasoning",
        "state": "All engineers in a company have technical training. Ravi is an engineer in the company.",
        "question": "Does Ravi have technical training?",
        "true_label": 1,
    },
    {
        "candidate_id": "H017",
        "domain": "reasoning",
        "state": "All engineers in a company have technical training. Ravi has technical training.",
        "question": "Does this prove that Ravi is an engineer?",
        "true_label": 0,
    },
    {
        "candidate_id": "H018",
        "domain": "reasoning",
        "state": "Some students in a class study mathematics. Ravi is a student in the class.",
        "question": "Does this prove that Ravi studies mathematics?",
        "true_label": 0,
    },
    {
        "candidate_id": "H019",
        "domain": "reasoning",
        "state": "No birds in the laboratory can fly. A robin is a bird in the laboratory.",
        "question": "Can the robin fly in the laboratory?",
        "true_label": 0,
    },
    {
        "candidate_id": "H020",
        "domain": "reasoning",
        "state": "If a machine overheats, an alarm activates. The alarm is activated.",
        "question": "Does this prove that the machine overheated?",
        "true_label": 0,
    },
    {
        "candidate_id": "H021",
        "domain": "reasoning",
        "state": "If a machine overheats, an alarm activates. The machine overheated.",
        "question": "Did the alarm activate?",
        "true_label": 1,
    },
    {
        "candidate_id": "H022",
        "domain": "reasoning",
        "state": "Every red card in a deck is a heart or diamond. Card X is red.",
        "question": "Must Card X be a heart?",
        "true_label": 0,
    },
    {
        "candidate_id": "H023",
        "domain": "reasoning",
        "state": "Every square is a rectangle. Shape A is a square.",
        "question": "Is Shape A a rectangle?",
        "true_label": 1,
    },
    {
        "candidate_id": "H024",
        "domain": "reasoning",
        "state": "Every rectangle is a quadrilateral. Shape A is a quadrilateral.",
        "question": "Must Shape A be a rectangle?",
        "true_label": 0,
    },

    # ---------------------------------------------------------
    # SCIENCE
    # ---------------------------------------------------------

    {
        "candidate_id": "H025",
        "domain": "science",
        "state": "Water is heated at standard atmospheric pressure.",
        "question": "Will the water eventually reach its boiling point?",
        "true_label": 1,
    },
    {
        "candidate_id": "H026",
        "domain": "science",
        "state": "A metal object is placed in water and sinks.",
        "question": "Is the object's average density greater than the density of water?",
        "true_label": 1,
    },
    {
        "candidate_id": "H027",
        "domain": "science",
        "state": "A plant receives sunlight, water, and carbon dioxide.",
        "question": "Can the plant perform photosynthesis?",
        "true_label": 1,
    },
    {
        "candidate_id": "H028",
        "domain": "science",
        "state": "A sealed container contains a gas. The gas is heated while its volume remains constant.",
        "question": "Does the gas pressure increase?",
        "true_label": 1,
    },
    {
        "candidate_id": "H029",
        "domain": "science",
        "state": "An object moves at constant velocity in a straight line.",
        "question": "Is the net force on the object zero?",
        "true_label": 1,
    },
    {
        "candidate_id": "H030",
        "domain": "science",
        "state": "An object is moving in a circle at constant speed.",
        "question": "Is the object's velocity constant?",
        "true_label": 0,
    },

    # ---------------------------------------------------------
    # GENERAL KNOWLEDGE / CONCEPTUAL
    # ---------------------------------------------------------

    {
        "candidate_id": "H031",
        "domain": "general_knowledge",
        "state": "A person travels from Delhi to Mumbai.",
        "question": "Must the person travel westward for the entire journey?",
        "true_label": 0,
    },
    {
        "candidate_id": "H032",
        "domain": "general_knowledge",
        "state": "A clock shows exactly 12:00 noon.",
        "question": "Must the Sun be directly overhead at the clock's location?",
        "true_label": 0,
    },
    {
        "candidate_id": "H033",
        "domain": "general_knowledge",
        "state": "A person is taller than 180 cm.",
        "question": "Is the person taller than 170 cm?",
        "true_label": 1,
    },
    {
        "candidate_id": "H034",
        "domain": "general_knowledge",
        "state": "A person is older than 20 years.",
        "question": "Is the person at least 30 years old?",
        "true_label": 0,
    },
    {
        "candidate_id": "H035",
        "domain": "general_knowledge",
        "state": "A box contains at least 10 objects.",
        "question": "Does the box contain exactly 10 objects?",
        "true_label": 0,
    },
    {
        "candidate_id": "H036",
        "domain": "general_knowledge",
        "state": "A box contains fewer than 10 objects.",
        "question": "Does the box contain at most 9 objects?",
        "true_label": 1,
    },

    # ---------------------------------------------------------
    # MULTI-STEP / BORDERLINE NUMERICAL REASONING
    # ---------------------------------------------------------

    {
        "candidate_id": "H037",
        "domain": "mathematics",
        "state": "A student answers 17 of 20 questions correctly.",
        "question": "Is the student's score greater than 80 percent?",
        "true_label": 1,
    },
    {
        "candidate_id": "H038",
        "domain": "mathematics",
        "state": "A student answers 16 of 20 questions correctly.",
        "question": "Is the student's score greater than 80 percent?",
        "true_label": 0,
    },
    {
        "candidate_id": "H039",
        "domain": "mathematics",
        "state": "A store has 80 items. It sells 31 items and receives 10 new items.",
        "question": "Does the store now have more than 60 items?",
        "true_label": 0,
    },
    {
        "candidate_id": "H040",
        "domain": "mathematics",
        "state": "A store has 80 items. It sells 19 items and receives 5 new items.",
        "question": "Does the store now have more than 60 items?",
        "true_label": 1,
    },
]


DATA.mkdir(parents=True, exist_ok=True)

fields = [
    "candidate_id",
    "domain",
    "state",
    "question",
    "true_label",
]

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(cases)


print("=" * 60)
print("E4b HARD CANDIDATE DATASET CREATED")
print("=" * 60)
print(f"Rows: {len(cases)}")
print(f"Output: {OUTPUT}")
print()

from collections import Counter

print("Domains:")
for k, v in Counter(
    row["domain"] for row in cases
).items():
    print(f"  {k}: {v}")

print()

print("Labels:")
for k, v in Counter(
    row["true_label"] for row in cases
).items():
    print(f"  {k}: {v}")

print()

print("Dataset ready for screening.")
