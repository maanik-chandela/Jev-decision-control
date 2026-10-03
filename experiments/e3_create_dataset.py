# ============================================================
# E3 DATASET — SELECTIVE PREDICTION / ABSTENTION
# ============================================================
#
# 100 fresh objectively answerable cases
# 5 domains × 20 cases
# 50 TRUE / 50 FALSE
# 10 TRUE / 10 FALSE per domain
#
# Difficulty:
#   easy     = 40
#   moderate = 40
#   hard     = 20
#
# Difficulty is a design variable only.
# It must NOT be interpreted as expected JEV confidence.
# ============================================================

from pathlib import Path
import pandas as pd


cases = []


def add(original_id, domain, difficulty, state, question, label):
    cases.append({
        "original_id": original_id,
        "domain": domain,
        "difficulty": difficulty,
        "state": state,
        "question": question,
        "label": label,
    })


# ============================================================
# MATHEMATICS — 20
# 10 TRUE / 10 FALSE
# ============================================================

math_cases = [
    ("M01", "easy", "A rectangle has length 8 cm and width 5 cm.",
     "Is its area 41 square centimeters?", 0),

    ("M02", "easy", "A triangle has base 10 cm and height 6 cm.",
     "Is its area 31 square centimeters?", 0),

    ("M03", "easy", "The number 26 is given.",
     "Is 26 a perfect square?", 0),

    ("M04", "easy", "A line has slope 2 and passes through (0, 3).",
     "Does the line pass through (1, 4)?", 0),

    ("M05", "easy", "A square has side length 7 cm.",
     "Is its perimeter 30 cm?", 0),

    ("M06", "moderate", "The equation is x + 9 = 14.",
     "Is x equal to 5?", 1),

    ("M07", "moderate", "The sequence is 2, 4, 6, 8, 10.",
     "Is the next term 12?", 1),

    ("M08", "moderate", "A circle has radius 3 cm.",
     "Is its diameter 6 cm?", 1),

    ("M09", "moderate", "The equation is 3x + 4 = 19.",
     "Is x equal to 5?", 1),

    ("M10", "moderate", "A number is increased from 80 to 100.",
     "Is the percentage increase 25 percent?", 1),

    ("M11", "moderate", "The average of 6, 8, 10, and 12 is considered.",
     "Is the average 9?", 1),

    ("M12", "moderate", "The quadratic equation is x^2 - 5x + 6 = 0.",
     "Is x = 2 one of its solutions?", 1),

    ("M13", "moderate", "A rectangle has area 72 square centimeters and width 8 cm.",
     "Is its length 9 cm?", 1),

    ("M14", "moderate", "The fraction 18/24 is simplified.",
     "Is its simplified value 2/3?", 0),

    ("M15", "hard", "A right triangle has legs 9 cm and 12 cm.",
     "Is its hypotenuse 15 cm?", 1),

    ("M16", "hard", "The equation 2x + 3 = 5x - 12 is given.",
     "Is x equal to 4?", 0),

    ("M17", "hard", "A number is first increased by 20 percent and then decreased by 20 percent.",
     "Does this sequence of changes return the number exactly to its original value?", 0),

    ("M18", "hard", "For positive a and b, a > b.",
     "Must a^2 be greater than b^2?", 1),

    ("M19", "easy", "The integer -7 is given.",
     "Is -7 greater than -3?", 0),

    ("M20", "moderate", "A number is 40 percent of 250.",
     "Is the number 120?", 0),
]


for item in math_cases:
    add(f"E3_{item[0]}", "mathematics", *item[1:])


# ============================================================
# PROBABILITY — 20
# 10 TRUE / 10 FALSE
# ============================================================

prob_cases = [
    ("P01", "easy", "A fair coin is tossed once.",
     "Is the probability of heads 0.5?", 1),

    ("P02", "easy", "A standard six-sided die is rolled once.",
     "Is the probability of rolling a 6 equal to 1/6?", 1),

    ("P03", "easy", "A standard six-sided die is rolled once.",
     "Is the probability of rolling an even number 1/2?", 1),

    ("P04", "easy", "A bag contains 3 red balls and 2 blue balls.",
     "Is the probability of drawing a red ball 3/5?", 1),

    ("P05", "easy", "A fair coin is tossed twice.",
     "Is the probability of getting two heads 1/4?", 1),

    ("P06", "easy", "A standard deck has 52 cards.",
     "Is the probability of drawing an ace 4/52?", 1),

    ("P07", "easy", "A die is rolled once.",
     "Is the probability of rolling a number greater than 4 equal to 1/3?", 1),

    ("P08", "easy", "A bag contains 5 green balls and 5 yellow balls.",
     "Is the probability of drawing a green ball 1/2?", 1),

    ("P09", "moderate", "A fair coin is tossed three times.",
     "Is the probability of getting exactly two heads 3/8?", 1),

    ("P10", "moderate", "A standard die is rolled twice.",
     "Is the probability that both rolls are 6 equal to 1/36?", 1),

    ("P11", "moderate", "A bag contains 4 red and 6 blue balls. One ball is drawn.",
     "Is the probability of drawing a red ball 0.5?", 0),

    ("P12", "moderate", "Two fair dice are rolled.",
     "Is the probability that their sum is 7 equal to 1/4?", 0),

    ("P13", "moderate", "A fair coin is tossed four times.",
     "Is the probability of getting at least one head 1/2?", 0),

    ("P14", "moderate", "A card is selected from a standard 52-card deck.",
     "Is the probability of selecting a heart 1/2?", 0),

    ("P15", "hard", "Two fair dice are rolled.",
     "Is the probability that the sum is greater than 10 equal to 1/6?", 0),

    ("P16", "hard", "A fair coin is tossed five times.",
     "Is the probability of getting exactly three heads 5/32?", 0),

    ("P17", "hard", "Two cards are drawn without replacement from a standard deck.",
     "Is the probability that both are aces equal to 1/52?", 0),

    ("P18", "hard", "A box contains 3 red and 7 blue balls. Two balls are drawn without replacement.",
     "Is the probability that both are red equal to 1/10?", 0),

    ("P19", "easy", "A fair six-sided die is rolled.",
     "Is the probability of rolling a 7 equal to 1/6?", 0),

    ("P20", "moderate", "A fair coin is tossed twice.",
     "Is the probability of getting at least one head equal to 1/4?", 0),
]


for item in prob_cases:
    add(f"E3_{item[0]}", "probability", *item[1:])


# ============================================================
# FORMAL LOGIC — 20
# 10 TRUE / 10 FALSE
# ============================================================

logic_cases = [
    ("L01", "easy", "All cats are mammals. Felix is a cat.",
     "Must Felix not be a mammal?", 0),

    ("L02", "easy", "All squares are rectangles. ABCD is a square.",
     "Must ABCD not be a rectangle?", 0),

    ("L03", "easy", "No birds are mammals. A sparrow is a bird.",
     "Must the sparrow be a mammal?", 0),

    ("L04", "easy", "All doctors are professionals. Ravi is a doctor.",
     "Must Ravi be a professional?", 1),

    ("L05", "easy", "Some students are athletes.",
     "Does this imply that at least one student is an athlete?", 1),

    ("L06", "easy", "No A are B. X is an A.",
     "Must X not be a B?", 1),

    ("L07", "moderate", "All poets are writers. Maya is a poet.",
     "Must Maya be a writer?", 1),

    ("L08", "easy", "Some engineers are musicians.",
     "Must every engineer be a musician?", 0),

    ("L09", "moderate", "If it rains, the ground becomes wet. It rains.",
     "Must the ground become wet?", 1),

    ("L10", "moderate", "If A occurs, then B occurs. B occurs.",
     "Must A have occurred?", 0),

    ("L11", "moderate", "All birds can fly. Penguins are birds.",
     "Does the premise set imply that penguins can fly?", 1),

    ("L12", "moderate", "Some A are B. No B are C.",
     "Must every A be outside C?", 0),

    ("L13", "moderate", "Some A are not B.",
     "Does this imply that at least one A exists?", 1),

    ("L14", "moderate", "All A are B. No B are C.",
     "Must every A be a non-C?", 1),

    ("L15", "hard", "If P then Q. If Q then R. P is true.",
     "Must R be true?", 1),

    ("L16", "hard", "If P then Q. Q is false.",
     "Must P be false?", 1),

    ("L17", "hard", "If P then Q. P is false.",
     "Must Q be false?", 0),

    ("L18", "hard", "All A are B. Some B are C.",
     "Must some A be C?", 0),

    ("L19", "easy", "No mammals are reptiles. A dog is a mammal.",
     "Must the dog be a reptile?", 0),

    ("L20", "moderate", "All A are B. All B are C.",
     "Must every C be an A?", 0),
]


for item in logic_cases:
    add(f"E3_{item[0]}", "formal_logic", *item[1:])


# ============================================================
# SCIENCE REASONING — 20
# 10 TRUE / 10 FALSE
# ============================================================

science_cases = [
    ("S01", "easy", "Water at standard atmospheric pressure boils at 100 degrees Celsius.",
     "Does water boil at 90 degrees Celsius under these conditions?", 0),

    ("S02", "easy", "An object has mass 2 kg and acceleration 3 m/s^2.",
     "Is the net force 5 newtons?", 0),

    ("S03", "easy", "Plants use sunlight as an energy source during photosynthesis.",
     "Does photosynthesis require no light energy?", 0),

    ("S04", "easy", "The Earth completes approximately one rotation in 24 hours.",
     "Does one rotation take approximately 12 hours?", 0),

    ("S05", "easy", "Sound requires a material medium to propagate.",
     "Can sound propagate through a perfect vacuum?", 0),

    ("S06", "easy", "A substance has pH 3.",
     "Is the substance neutral?", 0),

    ("S07", "moderate", "An object moves at constant velocity in a straight line.",
     "Is its acceleration zero?", 1),

    ("S08", "easy", "The density of an object is mass divided by volume.",
     "Is density equal to mass divided by volume?", 1),

    ("S09", "moderate", "A circuit has voltage 12 V and resistance 4 ohms.",
     "Is the current 3 A according to Ohm's law?", 1),

    ("S10", "moderate", "An isolated system receives no heat and performs no work on its surroundings.",
     "Does its total internal energy remain constant?", 1),

    ("S11", "moderate", "A metal wire is heated while its length is free to change.",
     "Does the wire generally expand when heated?", 1),

    ("S12", "moderate", "An object has density 2 g/cm^3 and volume 10 cm^3.",
     "Is its mass 10 g?", 0),

    ("S13", "moderate", "A force of 20 N acts on a mass of 5 kg.",
     "Is the resulting acceleration 4 m/s^2?", 1),

    ("S14", "moderate", "In a vacuum, two objects of different masses are dropped from the same height.",
     "Ignoring air resistance, do they have the same gravitational acceleration?", 1),

    ("S15", "hard", "A car increases its speed from 20 m/s to 30 m/s in 5 seconds.",
     "Is its average acceleration 2 m/s^2?", 1),

    ("S16", "hard", "A sealed object is electrically neutral.",
     "Must the object contain no charged particles?", 0),

    ("S17", "hard", "A gas is compressed while its temperature remains constant.",
     "Does its pressure generally increase according to Boyle's law?", 1),

    ("S18", "hard", "An object is moving in a circle at constant speed.",
     "Is its velocity necessarily constant?", 0),

    ("S19", "easy", "The Sun is closer to Earth than the Moon is.",
     "Is this statement true?", 0),

    ("S20", "moderate", "A solution has pH 7 at standard conditions.",
     "Is it approximately neutral?", 1),
]


for item in science_cases:
    add(f"E3_{item[0]}", "science_reasoning", *item[1:])


# ============================================================
# DATA REASONING — 20
# 10 TRUE / 10 FALSE
# ============================================================

data_cases = [
    ("D01", "easy",
     "A dataset contains the values 2, 4, 6, 8, and 10.",
     "Is the mean 7?", 0),

    ("D02", "easy",
     "A dataset contains the values 1, 2, 3, 4, and 5.",
     "Is the median 4?", 0),

    ("D03", "easy",
     "A dataset contains the values 5, 5, 7, 8, and 9.",
     "Is the mode 7?", 0),

    ("D04", "easy",
     "A classifier makes 90 correct predictions out of 100.",
     "Is its accuracy 80 percent?", 0),

    ("D05", "easy",
     "A binary classifier has TP=30, TN=50, FP=10, FN=10.",
     "Is its accuracy 70 percent?", 0),

    ("D06", "easy",
     "A binary classifier has TP=40 and FP=10.",
     "Is its precision 70 percent?", 0),

    ("D07", "easy",
     "A binary classifier has TP=45 and FN=5.",
     "Is its recall 80 percent?", 0),

    ("D08", "easy",
     "A dataset contains 20 positive and 80 negative examples.",
     "Is the positive-class proportion 20 percent?", 1),

    ("D09", "moderate",
     "A dataset contains 10, 20, 30, 40, and 100.",
     "Is the median 30?", 1),

    ("D10", "moderate",
     "A classifier has TP=50, TN=40, FP=5, FN=5.",
     "Is its accuracy 90 percent?", 1),

    ("D11", "moderate",
     "A classifier has TP=40 and FP=20.",
     "Is its precision 50 percent?", 0),

    ("D12", "moderate",
     "A classifier has TP=60 and FN=20.",
     "Is its recall 75 percent?", 1),

    ("D13", "moderate",
     "A dataset contains values 10, 10, 20, 30, and 30.",
     "Is the mean 20?", 1),

    ("D14", "moderate",
     "A dataset contains values 3, 5, 7, 9.",
     "Is the median 6?", 1),

    ("D15", "hard",
     "A classifier has TP=40, TN=50, FP=10, FN=0.",
     "Is its accuracy 90 percent?", 1),

    ("D16", "hard",
     "A classifier has TP=80 and FN=20.",
     "Is its recall 80 percent?", 1),

    ("D17", "hard",
     "A classifier has TP=40 and FP=20.",
     "Is its precision approximately 66.7 percent?", 1),

    ("D18", "hard",
     "A dataset contains 100 observations, of which 25 are missing.",
     "Is the missing-data proportion 20 percent?", 0),

    ("D19", "easy",
     "A dataset contains the values 2, 4, 6, 8.",
     "Is the mean 5?", 1),

    ("D20", "moderate",
     "A binary classifier has TP=30, TN=60, FP=10, FN=0.",
     "Is its accuracy 80 percent?", 0),
]


for item in data_cases:
    add(f"E3_{item[0]}", "data_reasoning", *item[1:])


# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(cases)


# ============================================================
# VALIDATION
# ============================================================

assert len(df) == 100, f"Expected 100 cases, got {len(df)}"

assert df["original_id"].is_unique, "Duplicate original_id found"

assert set(df["label"].unique()) == {0, 1}, \
    "Labels must contain both 0 and 1"

assert df["label"].sum() == 50, \
    f"Expected 50 TRUE labels, got {df['label'].sum()}"

assert (df["label"] == 0).sum() == 50, \
    f"Expected 50 FALSE labels, got {(df['label'] == 0).sum()}"

expected_domains = {
    "mathematics",
    "probability",
    "formal_logic",
    "science_reasoning",
    "data_reasoning",
}

assert set(df["domain"]) == expected_domains, \
    "Unexpected domain set"

domain_counts = df["domain"].value_counts()

assert all(domain_counts == 20), \
    f"Each domain must contain 20 cases:\n{domain_counts}"

domain_labels = pd.crosstab(df["domain"], df["label"])

for domain in expected_domains:
    assert domain_labels.loc[domain, 1] == 10, \
        f"{domain}: expected 10 TRUE cases"

    assert domain_labels.loc[domain, 0] == 10, \
        f"{domain}: expected 10 FALSE cases"


difficulty_counts = df["difficulty"].value_counts()

assert difficulty_counts["easy"] == 40, \
    f"Expected 40 easy cases, got {difficulty_counts['easy']}"

assert difficulty_counts["moderate"] == 40, \
    f"Expected 40 moderate cases, got {difficulty_counts['moderate']}"

assert difficulty_counts["hard"] == 20, \
    f"Expected 20 hard cases, got {difficulty_counts['hard']}"


# ============================================================
# SAVE
# ============================================================

output_path = Path(
    "experiments/data/e3_selective_prediction_dataset.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)


# ============================================================
# FINAL REPORT
# ============================================================

print("=" * 60)
print("E3 DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Rows: {len(df)}")

print("\nLabels:")
print(df["label"].value_counts().sort_index())

print("\nDomains:")
print(df["domain"].value_counts().sort_index())

print("\nLabels by domain:")
print(pd.crosstab(df["domain"], df["label"]))

print("\nDifficulty:")
print(df["difficulty"].value_counts().sort_index())

print("\nOutput:")
print(output_path)

print("\nAll validation checks passed.")