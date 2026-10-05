import pandas as pd
from pathlib import Path

# E10 — COMPUTATION ALLOCATION DATASET
# 150 objectively answerable binary decisions.
# 5 domains x 30; 10 easy + 10 moderate + 10 hard per domain.
# 75 TRUE / 75 FALSE overall; 15 TRUE / 15 FALSE per domain.
# label: 1 = TRUE, 0 = FALSE

CASES = []

def add(domain, case_id, difficulty, question, label):
    CASES.append({
        "case_id": case_id,
        "domain": domain,
        "difficulty": difficulty,
        "question": question,
        "label": int(label),
    })


def add_rows(domain, rows):
    for row in rows:
        add(domain, *row)

add_rows("mathematics", [
    ("M01", "easy", "Is 17 greater than 20?", 0),
    ("M02", "easy", "Is 8 multiplied by 7 equal to 56?", 1),
    ("M03", "easy", "Is 144 divisible by 11?", 0),
    ("M04", "easy", "Is the square root of 81 equal to 9?", 1),
    ("M05", "easy", "Is 3/4 less than 1/2?", 0),
    ("M06", "easy", "Is 25 an odd number?", 1),
    ("M07", "easy", "Is 81 the square of 8?", 0),
    ("M08", "easy", "Is 2^5 equal to 32?", 1),
    ("M09", "easy", "Is 49 divisible by 7?", 1),
    ("M10", "easy", "Is the integer 36 divisible by 5?", 0),
    ("M11", "moderate", "A rectangle has length 12 cm and width 5 cm. Is its area 70 square centimeters?", 0),
    ("M12", "moderate", "A square has side length 6 cm. Is its perimeter 24 cm?", 1),
    ("M13", "moderate", "If x + 7 = 19, is x equal to 12?", 1),
    ("M14", "moderate", "Is the average of 8, 12, and 16 equal to 13?", 0),
    ("M15", "moderate", "If 3x = 21, is x equal to 8?", 0),
    ("M16", "moderate", "Is the area of a triangle with base 10 cm and height 6 cm equal to 30 square centimeters?", 1),
    ("M17", "moderate", "If x = 5, is 2x + 3 equal to 14?", 0),
    ("M18", "moderate", "Is 15% of 200 equal to 30?", 1),
    ("M19", "moderate", "If 5x - 10 = 15, is x equal to 5?", 1),
    ("M20", "moderate", "Is the circumference of a circle with radius 1 exactly 2π?", 1),
    ("M21", "hard", "If an integer is divisible by both 6 and 10, must it be divisible by 20?", 0),
    ("M22", "hard", "If a number is divisible by 12, must it be divisible by 8?", 0),
    ("M23", "hard", "For every real x, if x > 2 then x^2 > 4?", 1),
    ("M24", "hard", "For every real x, is x^2 greater than or equal to x?", 0),
    ("M25", "hard", "If a and b are positive and a > b, must a^2 < b^2?", 0),
    ("M26", "hard", "If two real numbers have the same square, must the two numbers be equal?", 0),
    ("M27", "hard", "If x and y are positive real numbers and x > y, must x^2 > y^2?", 1),
    ("M28", "hard", "If an integer is divisible by 4 and 6, must it be divisible by 12?", 1),
    ("M29", "hard", "If n^2 is even, must n be even?", 1),
    ("M30", "hard", "If a^2 = b^2 for real numbers a and b, must a = b?", 0),
])

add_rows("probability", [
    ("P01", "easy", "For a fair six-sided die, is the probability of rolling a 6 equal to 1/5?", 0),
    ("P02", "easy", "For a fair coin, is the probability of heads equal to 1/3?", 0),
    ("P03", "easy", "For a fair six-sided die, is the probability of rolling a number greater than 6 equal to 1/6?", 0),
    ("P04", "easy", "For a fair six-sided die, is the probability of rolling an even number equal to 1/2?", 1),
    ("P05", "easy", "Is the probability of an impossible event equal to 0?", 1),
    ("P06", "easy", "Is the probability of a certain event equal to 1?", 1),
    ("P07", "easy", "For a fair coin, is the probability of tails equal to 1/3?", 0),
    ("P08", "easy", "For a fair six-sided die, is the probability of rolling a number less than 7 equal to 1?", 1),
    ("P09", "easy", "In a finite sample space, if an event has probability 0, is the event impossible?", 1),
    ("P10", "easy", "For a fair six-sided die, is the probability of rolling 2 or 5 equal to 1/2?", 0),
    ("P11", "moderate", "For a fair six-sided die, is the probability of rolling a number greater than 4 equal to 1/3?", 1),
    ("P12", "moderate", "For a fair six-sided die, is the probability of rolling a prime number equal to 1/3?", 0),
    ("P13", "moderate", "If two events are mutually exclusive, can both occur in the same trial?", 0),
    ("P14", "moderate", "For two independent fair coin tosses, is the probability of getting two heads equal to 1/4?", 1),
    ("P15", "moderate", "For two independent fair coin tosses, is the probability of getting at least one head equal to 1/2?", 0),
    ("P16", "moderate", "A bag contains 3 red balls and 2 blue balls. Is the probability of drawing a red ball equal to 3/5?", 1),
    ("P17", "moderate", "A bag contains 4 red balls and 6 blue balls. Is the probability of drawing a red ball equal to 2/5?", 1),
    ("P18", "moderate", "If P(A) = 0.4, must P(A complement) equal 0.4?", 0),
    ("P19", "moderate", "If P(A) = 0.3 and P(B) = 0.2 for mutually exclusive events, is P(A or B) = 0.5?", 1),
    ("P20", "moderate", "For two fair dice, is the probability that their sum is 2 equal to 1/6?", 0),
    ("P21", "hard", "For two fair six-sided dice, is the probability that the sum is 7 equal to 1/5?", 0),
    ("P22", "hard", "For two fair six-sided dice, is the probability that the sum is greater than 10 equal to 1/12?", 1),
    ("P23", "hard", "If events A and B are independent, must P(A and B) equal P(A) + P(B)?", 0),
    ("P24", "hard", "If P(A) = 0.5 and P(B) = 0.5 and A and B are independent, is P(A and B) = 0.25?", 1),
    ("P25", "hard", "If P(A|B) = P(A), does that establish independence of A and B when P(B) > 0?", 1),
    ("P26", "hard", "For two fair six-sided dice, is the probability that the second die is greater than the first equal to 1/2?", 0),
    ("P27", "hard", "If A is a subset of B, must P(A) be less than or equal to P(B)?", 1),
    ("P28", "hard", "If P(A) = 0.7 and P(B) = 0.6, must P(A and B) equal 0.42?", 0),
    ("P29", "hard", "If P(A) = 0.8 and P(B) = 0.5, can P(A and B) be 0.4?", 1),
    ("P30", "hard", "If P(A or B) = P(A) + P(B), must A and B be independent?", 0),
])

add_rows("formal_logic", [
    ("L01", "easy", "If all cats are animals, and Luna is a cat, must Luna be a plant?", 0),
    ("L02", "easy", "If all birds can fly, and Robin is a bird, does the premise imply that Robin can fly?", 1),
    ("L03", "easy", "If no A are B, and x is an A, must x be a B?", 0),
    ("L04", "easy", "If all A are B, and x is not B, must x be an A?", 0),
    ("L05", "easy", "If A implies B, and A is true, must B be true?", 1),
    ("L06", "easy", "If A implies B, and B is false, must A be true?", 0),
    ("L07", "easy", "If all dogs are mammals and Rex is a dog, must Rex be a mammal?", 1),
    ("L08", "easy", "If no students are teachers and Ravi is a student, must Ravi be a teacher?", 0),
    ("L09", "easy", "If all roses are flowers and some roses exist, must at least one flower exist?", 1),
    ("L10", "easy", "If all programmers are adults and Ravi is not an adult, can Ravi be a programmer under these premises?", 0),
    ("L11", "moderate", "If every bird in a specified group is an animal and X is not an animal, must X not be a bird in that group?", 1),
    ("L12", "moderate", "If a device is certified, then it passes inspection. Device X is certified. Must Device X fail inspection?", 0),
    ("L13", "moderate", "If no A are B and some C are A, must those C that are A be non-B?", 1),
    ("L14", "moderate", "If every certified device passes inspection and device X is certified, must device X fail inspection?", 0),
    ("L15", "moderate", "If all A are B and all B are C, must all A be C?", 1),
    ("L16", "moderate", "If all A are B and some B are C, must some A be C?", 0),
    ("L17", "moderate", "If some A are B and no B are C, must every A be non-C?", 0),
    ("L18", "moderate", "If no A are B and all C are A, must no C be B?", 1),
    ("L19", "moderate", "If A implies B and B implies C, does A imply C?", 1),
    ("L20", "moderate", "If some A are B and no B are C, does it follow that every A is non-C?", 0),
    ("L21", "hard", "If every A is B, every B is C, and no C is D, must every A be non-D?", 1),
    ("L22", "hard", "If every A is B and no B is C, can an A also be a C under these premises?", 0),
    ("L23", "hard", "If some A are B and every B is C, must at least one A be C?", 1),
    ("L24", "hard", "If every A is B and some B are C, does it follow that some A are C?", 0),
    ("L25", "hard", "If no A are B, and every C is A, must no C be B?", 1),
    ("L26", "hard", "If A implies B, B implies C, and C is false, must A be false?", 1),
    ("L27", "hard", "If A implies B and B is true, must A be true?", 0),
    ("L28", "hard", "If A and B cannot both be true, and A is true, must B be false?", 1),
    ("L29", "hard", "If at least one A is B and every B is C, must at least one A be C?", 1),
    ("L30", "hard", "If at least one A is B, and A is false, must B be false?", 0),
])

add_rows("science_reasoning", [
    ("S01", "easy", "Does water freeze at 0 degrees Celsius under standard atmospheric pressure?", 1),
    ("S02", "easy", "Does the Earth orbit the Sun?", 1),
    ("S03", "easy", "Is the Earth a star?", 0),
    ("S04", "easy", "Is oxygen a gas at standard room conditions?", 1),
    ("S05", "easy", "Is oxygen required for ordinary anaerobic respiration?", 0),
    ("S06", "easy", "Does the Moon produce its own visible light like a star?", 0),
    ("S07", "easy", "Is liquid water made of H2O molecules?", 1),
    ("S08", "easy", "Does increasing temperature generally increase the average kinetic energy of particles in a substance?", 1),
    ("S09", "easy", "Is gravity a force that repels masses from one another?", 0),
    ("S10", "easy", "Can sound propagate through a vacuum?", 0),
    ("S11", "moderate", "Does light travel faster in vacuum than in air?", 1),
    ("S12", "moderate", "Does sound travel through a vacuum?", 0),
    ("S13", "moderate", "If the net force on an object is zero, must the object be stationary?", 0),
    ("S14", "moderate", "If the net force on an object is zero, can the object move at constant velocity?", 1),
    ("S15", "moderate", "For a fixed mass, does increasing net force increase acceleration?", 1),
    ("S16", "moderate", "If two objects have the same mass and experience the same net force, must their accelerations be equal?", 1),
    ("S17", "moderate", "Does an object with zero velocity necessarily have zero acceleration?", 0),
    ("S18", "moderate", "If an object is accelerating, must its speed be increasing?", 0),
    ("S19", "moderate", "Does increasing the frequency of a sound wave while keeping its propagation medium fixed increase its wavelength?", 0),
    ("S20", "moderate", "At constant speed in a circle, is the object still accelerating?", 1),
    ("S21", "hard", "If two objects experience the same net force but have different masses, must their accelerations be different?", 1),
    ("S22", "hard", "If two objects experience the same force and have different masses, must their accelerations be equal?", 0),
    ("S23", "hard", "If an object has constant momentum and constant mass, must its velocity be constant?", 1),
    ("S24", "hard", "Can an object have zero acceleration while moving with nonzero constant velocity?", 1),
    ("S25", "hard", "If two objects have equal kinetic energy, must they have equal momentum?", 0),
    ("S26", "hard", "If two objects have equal momentum, must they have equal kinetic energy?", 0),
    ("S27", "hard", "If an isolated system has no external net force, must its total momentum be conserved?", 1),
    ("S28", "hard", "If an object has nonzero acceleration, must its speed be changing?", 0),
    ("S29", "hard", "For a fixed mass moving in a circle at constant speed, changing the radius changes the required centripetal acceleration.", 1),
    ("S30", "hard", "If the net work done on an object is zero, must the object's speed have remained constant throughout the motion?", 0),
])

add_rows("data_reasoning", [
    ("D01", "easy", "A dataset contains the values 2, 4, 6, 8. Is the mean equal to 6?", 0),
    ("D02", "easy", "A dataset contains the values 1, 3, 5, 7, 9. Is the median equal to 6?", 0),
    ("D03", "easy", "A model correctly classifies 95 out of 100 examples. Is its accuracy 95%?", 1),
    ("D04", "easy", "A confusion matrix has TP=80 and FN=20. Is recall 70%?", 0),
    ("D05", "easy", "A classifier has precision 1.0. Does this necessarily mean its recall is also 1.0?", 0),
    ("D06", "easy", "A dataset contains 90 negative examples and 10 positive examples. Are the classes balanced?", 0),
    ("D07", "easy", "A sample contains the values 10, 10, 10, 10, and 20. Is the median equal to 10?", 1),
    ("D08", "easy", "A correlation coefficient between X and Y is exactly zero. Does this prove that X and Y are independent?", 0),
    ("D09", "easy", "A scatter plot shows a perfect straight-line relationship with positive slope. Is the Pearson correlation coefficient equal to 1?", 1),
    ("D10", "easy", "A sample has standard deviation zero. Must every observation have the same value?", 1),
    ("D11", "moderate", "A confusion matrix has TP=50, FP=50, FN=0, TN=100. Is recall equal to 100%?", 1),
    ("D12", "moderate", "A confusion matrix has TP=50, FP=50, FN=0, TN=100. Is precision equal to 50%?", 1),
    ("D13", "moderate", "A model's test accuracy is 99% on a dataset where 99% of examples belong to one class. Does 99% accuracy alone establish that the model handles both classes well?", 0),
    ("D14", "moderate", "Five measurements are 5, 10, 10, 20, and 25. Is the median equal to 10?", 1),
    ("D15", "moderate", "A dataset contains 100 observations, but 25 are missing from the recorded table. Does the recorded table contain 100 observed values?", 0),
    ("D16", "moderate", "If TP=80 and FP=20, is precision equal to 70%?", 0),
    ("D17", "moderate", "If TP=80 and FN=20, is recall equal to 80%?", 1),
    ("D18", "moderate", "If a dataset has mean 10 and every observation is increased by 5, does the new mean become 15?", 1),
    ("D19", "moderate", "If every value in a dataset is multiplied by 2, does the median remain unchanged?", 0),
    ("D20", "moderate", "A dataset contains 10, 20, and 30. Is its mean equal to 20?", 1),
    ("D21", "hard", "If a classifier has precision 1.0, must every predicted positive be a true positive?", 1),
    ("D22", "hard", "If a classifier has recall 1.0, must every predicted positive be a true positive?", 0),
    ("D23", "hard", "If a model has 90% accuracy on a dataset with 90% negatives, does that alone prove it learned meaningful patterns?", 0),
    ("D24", "hard", "If a correlation coefficient is 1, must the relationship between the two variables be exactly linear with positive slope?", 1),
    ("D25", "hard", "If two datasets have the same mean, must they have the same variance?", 0),
    ("D26", "hard", "For the dataset [1, 2, 3, 100], is the mean equal to 26.5 and the median equal to 2.5?", 1),
    ("D27", "hard", "If every observation in a dataset is increased by the same constant, does the variance remain unchanged?", 1),
    ("D28", "hard", "If two models have the same accuracy, must they have the same confusion matrix?", 0),
    ("D29", "hard", "If a classifier has zero false positives, must its precision be 100% provided it makes at least one positive prediction?", 1),
    ("D30", "hard", "If every value in a dataset is multiplied by 3, does the median remain unchanged?", 0),
])

# Build dataframe.
df = pd.DataFrame(CASES)

# Structural validation.
assert len(df) == 150, f"Expected 150 cases, got {len(df)}"
assert df["case_id"].is_unique, "Case IDs are not unique"
assert set(df["label"].unique()) == {0, 1}, "Labels must be exactly 0 and 1"

label_counts = df["label"].value_counts().sort_index()
assert label_counts[0] == 75, f"Expected 75 false cases, got {label_counts[0]}"
assert label_counts[1] == 75, f"Expected 75 true cases, got {label_counts[1]}"

expected_domains = {
    "mathematics": 30,
    "probability": 30,
    "formal_logic": 30,
    "science_reasoning": 30,
    "data_reasoning": 30,
}
assert df["domain"].value_counts().to_dict() == expected_domains

expected_difficulties = {"easy": 50, "moderate": 50, "hard": 50}
assert df["difficulty"].value_counts().to_dict() == expected_difficulties

cross = pd.crosstab(df["domain"], df["difficulty"])
for domain in expected_domains:
    assert cross.loc[domain, "easy"] == 10
    assert cross.loc[domain, "moderate"] == 10
    assert cross.loc[domain, "hard"] == 10

for domain in expected_domains:
    counts = df.loc[df["domain"] == domain, "label"].value_counts()
    assert counts[0] == 15, f"{domain}: expected 15 false"
    assert counts[1] == 15, f"{domain}: expected 15 true"

assert df["question"].is_unique, "Duplicate question text detected"

# Save dataset.
OUTPUT = Path("experiments/data/e10_computation_allocation_dataset.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("E10 COMPUTATION ALLOCATION DATASET")
print("=" * 60)
print(f"Total cases: {len(df)}")
print("\nLABELS:")
print(df["label"].value_counts().sort_index())
print("\nDOMAINS:")
print(df["domain"].value_counts().sort_index())
print("\nDIFFICULTY:")
print(df["difficulty"].value_counts().sort_index())
print("\nDOMAIN × DIFFICULTY:")
print(cross)
print("\nDOMAIN × LABEL:")
print(pd.crosstab(df["domain"], df["label"]))
print(f"\nSaved to: {OUTPUT}")
print("\nValidation: PASSED")
print("=" * 60)
