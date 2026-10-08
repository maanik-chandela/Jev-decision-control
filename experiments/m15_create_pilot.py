from pathlib import Path
import csv

OUT = Path("experiments/data/m15_pilot.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

cases = [
    # Mathematics
    ("M01", "math", "easy", "If x = 5, then x + 3 = 8.", 1),
    ("M02", "math", "easy", "If 7 is greater than 10, then 7 is greater than 5.", 0),
    ("M03", "math", "easy", "The square of every real number is nonnegative.", 1),
    ("M04", "math", "easy", "If a number is divisible by 4, then it is divisible by 2.", 1),
    ("M05", "math", "easy", "If x > 5, then x > 10.", 0),
    ("M06", "math", "easy", "The average of 4 and 8 is 6.", 1),

    # Probability
    ("P01", "probability", "easy", "A fair coin has probability 1/2 of landing heads on one toss.", 1),
    ("P02", "probability", "easy", "The probability of rolling a 7 on a standard six-sided die is 1/6.", 0),
    ("P03", "probability", "easy", "The probability of an impossible event is 0.", 1),
    ("P04", "probability", "easy", "The probability of an event can be greater than 1.", 0),
    ("P05", "probability", "easy", "For independent events A and B, P(A and B) = P(A)P(B).", 1),
    ("P06", "probability", "easy", "A probability of 0.25 is equal to 25%.", 1),

    # Formal logic
    ("L01", "formal_logic", "easy", "If all cats are mammals and all mammals are animals, then all cats are animals.", 1),
    ("L02", "formal_logic", "easy", "If P implies Q and Q is false, then P must be true.", 0),
    ("L03", "formal_logic", "easy", "If P is true and P implies Q, then Q is true.", 1),
    ("L04", "formal_logic", "easy", "If all A are B, then every B must be an A.", 0),
    ("L05", "formal_logic", "easy", "A contradiction cannot be both true and false under classical logic.", 1),
    ("L06", "formal_logic", "easy", "If P and Q are both true, then P or Q is true.", 1),

    # Science reasoning
    ("S01", "science_reasoning", "easy", "Water freezes at 0 degrees Celsius under standard atmospheric pressure.", 1),
    ("S02", "science_reasoning", "easy", "The Earth is the closest planet to the Sun.", 0),
    ("S03", "science_reasoning", "easy", "Plants use photosynthesis to convert light energy into chemical energy.", 1),
    ("S04", "science_reasoning", "easy", "Sound can travel through a perfect vacuum.", 0),
    ("S05", "science_reasoning", "easy", "Gravity attracts masses toward one another.", 1),
    ("S06", "science_reasoning", "easy", "The human heart normally has four chambers.", 1),

    # Data reasoning
    ("D01", "data_reasoning", "easy", "A dataset containing 10, 20, and 30 has a mean of 20.", 1),
    ("D02", "data_reasoning", "easy", "If every value in a dataset increases by 5, the mean also increases by 5.", 1),
    ("D03", "data_reasoning", "easy", "A correlation coefficient of 0 means the two variables must have a causal relationship.", 0),
    ("D04", "data_reasoning", "easy", "If 8 out of 10 observations are correct, the accuracy is 80%.", 1),
    ("D05", "data_reasoning", "easy", "A median must always be larger than the mean.", 0),
    ("D06", "data_reasoning", "easy", "If a classifier makes 2 errors out of 100 predictions, its accuracy is 98%.", 1),
]

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "domain", "difficulty", "question", "label"])
    writer.writerows(cases)

print(f"Created {OUT}")
print(f"Cases: {len(cases)}")
print(f"True: {sum(x[4] for x in cases)}")
print(f"False: {sum(1 - x[4] for x in cases)}")
