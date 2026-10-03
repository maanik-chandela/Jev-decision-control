from pathlib import Path
import csv

OUT = Path("experiments/data/e4c_h_candidates.csv")


def add(rows, case_id, domain, state, question, label, difficulty="hard"):
    rows.append({
        "id": case_id,
        "domain": domain,
        "state": state,
        "question": question,
        "label": label,
        "difficulty": difficulty,
    })


rows = []

# ============================================================
# MATHEMATICS — 20
# ============================================================

math_cases = [
    (
        "The equation 3x + 7 = 25 has a real solution x = 6.",
        "Is x = 6 the solution?",
        1,
    ),
    (
        "A rectangle has length 12 cm and width 5 cm. Its area is 50 square centimeters.",
        "Is the area 50 square centimeters?",
        0,
    ),
    (
        "A number is divisible by 6. Every number divisible by 6 is divisible by 3.",
        "Must the number be divisible by 3?",
        1,
    ),
    (
        "The average of five numbers is 18. Four of the numbers are 12, 16, 20, and 22.",
        "Is the fifth number 20?",
        0,
    ),
    (
        "A triangle has angles of 35 degrees and 65 degrees.",
        "Is the third angle 80 degrees?",
        1,
    ),
    (
        "For a positive number x, x^2 = 49.",
        "Must x equal 7?",
        1,
    ),
    (
        "A fair six-sided die is rolled twice.",
        "Is the probability of obtaining two sixes exactly 1/36?",
        1,
    ),
    (
        "A bag contains 4 red balls and 6 blue balls. One ball is selected uniformly at random.",
        "Is the probability of selecting a red ball 2/5?",
        1,
    ),
    (
        "A sequence is defined by a_n = 2n + 3.",
        "Is a_10 equal to 23?",
        1,
    ),
    (
        "The function f(x) = x^2 is evaluated at x = -4.",
        "Is f(-4) equal to -16?",
        0,
    ),
    (
        "A class contains 30 students. 18 students study mathematics and 15 study physics. "
        "There are 5 students who study both.",
        "Are there exactly 28 students who study at least one of the two subjects?",
        1,
    ),
    (
        "A price is increased by 20% and then decreased by 20%.",
        "Is the final price equal to the original price?",
        0,
    ),
    (
        "The equation x^2 - 5x + 6 = 0 has roots 2 and 3.",
        "Are 2 and 3 both solutions of the equation?",
        1,
    ),
    (
        "A right triangle has legs of length 6 and 8.",
        "Is its hypotenuse 10?",
        1,
    ),
    (
        "For x > 0, log_10(x) = 2.",
        "Is x equal to 100?",
        1,
    ),
    (
        "A set contains 5 distinct elements.",
        "Does its power set contain exactly 10 elements?",
        0,
    ),
    (
        "The determinant of a 2x2 matrix is zero.",
        "Must the matrix be invertible?",
        0,
    ),
    (
        "For every real number x, x^2 is nonnegative.",
        "Can x^2 ever be negative?",
        0,
    ),
    (
        "A fair coin is tossed three times.",
        "Is the probability of getting exactly two heads equal to 3/8?",
        1,
    ),
    (
        "A population doubles every 5 years. Its initial size is 200.",
        "After 10 years, is the population 800?",
        1,
    ),
]

for i, (state, question, label) in enumerate(math_cases, 1):
    add(rows, f"E4CH_M{i:02d}", "mathematics", state, question, label)


# ============================================================
# PROBABILITY / QUANTITATIVE REASONING — 15
# ============================================================

prob_cases = [
    (
        "A box contains 3 red, 4 blue, and 5 green balls. Two balls are drawn "
        "without replacement.",
        "Is the probability that both balls are red equal to 1/22?",
        1,
    ),
    (
        "A fair coin is tossed repeatedly.",
        "Is the probability that the first two tosses are both heads equal to 1/4?",
        1,
    ),
    (
        "A test has 4 independent questions, each with probability 1/2 of being answered correctly.",
        "Is the expected number of correct answers 2?",
        1,
    ),
    (
        "A random variable X takes values 0 and 2 with equal probability.",
        "Is the expected value of X equal to 1?",
        1,
    ),
    (
        "A population has 40% individuals with property A and 30% with property B. "
        "The events are mutually exclusive.",
        "Is the probability of A or B equal to 70%?",
        1,
    ),
    (
        "Events A and B are independent, with P(A)=0.4 and P(B)=0.5.",
        "Is P(A and B) equal to 0.9?",
        0,
    ),
    (
        "A fair die is rolled once.",
        "Is the expected value of the outcome 3.5?",
        1,
    ),
    (
        "A company has 60 employees. 40% work remotely.",
        "Is the number of remote employees 24?",
        1,
    ),
    (
        "A quantity increases from 80 to 100.",
        "Is this an increase of exactly 20%?",
        1,
    ),
    (
        "A quantity decreases from 100 to 80.",
        "Is this a decrease of exactly 20%?",
        1,
    ),
    (
        "A machine produces 2% defective items independently. Ten items are produced.",
        "Is the expected number of defective items 0.2?",
        1,
    ),
    (
        "A survey samples 100 people and 60 answer yes.",
        "Does this prove that exactly 60% of the entire population would answer yes?",
        0,
    ),
    (
        "A fair six-sided die is rolled twice.",
        "Is the probability that the sum is 7 equal to 1/6?",
        1,
    ),
    (
        "A bag contains 2 red and 8 blue balls. A ball is drawn, replaced, and another ball is drawn.",
        "Are the two draws independent?",
        1,
    ),
    (
        "A probability model assigns probabilities 0.2, 0.3, and 0.4 to three mutually exclusive outcomes.",
        "Can these three probabilities describe the complete probability distribution?",
        0,
    ),
]

for i, (state, question, label) in enumerate(prob_cases, 1):
    add(rows, f"E4CH_P{i:02d}", "probability", state, question, label)


# ============================================================
# FORMAL / CONDITIONAL LOGIC — 15
# ============================================================

logic_cases = [
    (
        "Every A is a B. Every B is a C.",
        "Must every A be a C?",
        1,
    ),
    (
        "Every A is a B. Some B are not C.",
        "Must some A be not C?",
        0,
    ),
    (
        "No A is a B. x is an A.",
        "Can x also be a B?",
        0,
    ),
    (
        "If a number is divisible by 4, then it is even. The number n is not even.",
        "Can n be divisible by 4?",
        0,
    ),
    (
        "If P then Q. Q is true.",
        "Must P be true?",
        0,
    ),
    (
        "If P then Q. P is false.",
        "Must Q be false?",
        0,
    ),
    (
        "If P then Q. Q is false.",
        "Must P be false?",
        1,
    ),
    (
        "If P then Q. P is true.",
        "Must Q be true?",
        1,
    ),
    (
        "All engineers in a group know Python. Ravi is an engineer in the group.",
        "Must Ravi know Python?",
        1,
    ),
    (
        "Some students study mathematics. Every mathematics student studies algebra.",
        "Must at least one student study algebra?",
        1,
    ),
    (
        "No mammals lay eggs. The platypus is a mammal.",
        "Must the platypus not lay eggs?",
        1,
    ),
    (
        "Every red object is heavy. An object is not heavy.",
        "Must the object not be red?",
        1,
    ),
    (
        "Some A are B. Some B are C.",
        "Must some A also be C?",
        0,
    ),
    (
        "Every A is B. No B is C.",
        "Can an object be both A and C?",
        0,
    ),
    (
        "If the server is online, the website responds. The website does not respond.",
        "Must the server be offline?",
        1,
    ),
]

for i, (state, question, label) in enumerate(logic_cases, 1):
    add(rows, f"E4CH_L{i:02d}", "formal_logic", state, question, label)


# ============================================================
# SCIENCE REASONING — 15
# ============================================================

science_cases = [
    (
        "At constant pressure, an ideal gas is heated from 300 K to 600 K.",
        "Does its volume double according to Charles's law?",
        1,
    ),
    (
        "An object moves at constant velocity for 10 seconds.",
        "Is its acceleration zero during this interval?",
        1,
    ),
    (
        "A closed system receives 500 J of heat and performs 200 J of work.",
        "Is its internal energy change 300 J under the convention ΔU = Q - W?",
        1,
    ),
    (
        "A solution has pH 3.",
        "Is it acidic?",
        1,
    ),
    (
        "An object has mass 2 kg and acceleration 3 m/s^2.",
        "Is the net force 6 N?",
        1,
    ),
    (
        "Light travels from air into glass.",
        "Does its frequency decrease because its speed decreases?",
        0,
    ),
    (
        "A reaction has a positive activation energy.",
        "Does increasing temperature generally increase the fraction of molecules "
        "with energy above the activation barrier?",
        1,
    ),
    (
        "A neutral atom loses one electron.",
        "Does it become a negatively charged ion?",
        0,
    ),
    (
        "At constant temperature, the pressure of an ideal gas is inversely proportional "
        "to its volume.",
        "If the volume doubles, does the pressure become half its original value?",
        1,
    ),
    (
        "An isolated object experiences no net external force.",
        "Must its momentum remain constant?",
        1,
    ),
    (
        "A radioactive sample has a half-life of 10 years.",
        "After 20 years, is 25% of the original quantity expected to remain?",
        1,
    ),
    (
        "Water is heated from 20°C to 80°C at constant pressure.",
        "Does its temperature increase by 300°C?",
        0,
    ),
    (
        "A catalyst lowers the activation energy of both the forward and reverse reactions.",
        "Does a catalyst change the equilibrium constant of the reaction?",
        0,
    ),
    (
        "An electron and a proton are placed in the same uniform electric field.",
        "Do they experience electric forces in opposite directions?",
        1,
    ),
    (
        "A metal wire is heated while its material remains unchanged.",
        "Does its electrical resistance generally increase for an ordinary metal?",
        1,
    ),
]

for i, (state, question, label) in enumerate(science_cases, 1):
    add(rows, f"E4CH_S{i:02d}", "science_reasoning", state, question, label)


# ============================================================
# DATA / STRUCTURED REASONING — 15
# ============================================================

data_cases = [
    (
        "A dataset contains values 2, 4, 6, 8, and 10.",
        "Is the mean equal to 6?",
        1,
    ),
    (
        "A dataset contains values 1, 1, 1, 9.",
        "Is the median equal to 1?",
        1,
    ),
    (
        "A confusion matrix has TP=80, FP=20, FN=10, TN=90.",
        "Is precision equal to 80%?",
        0,
    ),
    (
        "A classifier has TP=80 and FN=20.",
        "Is its recall 80%?",
        1,
    ),
    (
        "A classifier has precision 1.0.",
        "Does this necessarily mean its recall is also 1.0?",
        0,
    ),
    (
        "A dataset contains 90 negative examples and 10 positive examples.",
        "Are the classes balanced?",
        0,
    ),
    (
        "A model correctly classifies 95 out of 100 examples.",
        "Is its accuracy 95%?",
        1,
    ),
    (
        "A correlation coefficient between X and Y is exactly zero.",
        "Does this prove that X and Y are independent?",
        0,
    ),
    (
        "A scatter plot shows a perfect straight-line relationship with positive slope.",
        "Is the Pearson correlation coefficient equal to 1?",
        1,
    ),
    (
        "A sample has standard deviation zero.",
        "Must every observation have the same value?",
        1,
    ),
    (
        "A confusion matrix has TP=50, FP=50, FN=0, TN=100.",
        "Is recall equal to 100%?",
        1,
    ),
    (
        "A confusion matrix has TP=50, FP=50, FN=0, TN=100.",
        "Is precision equal to 50%?",
        1,
    ),
    (
        "A model's test accuracy is 99% on a dataset where 99% of examples belong "
        "to one class.",
        "Does 99% accuracy alone establish that the model handles both classes well?",
        0,
    ),
    (
        "Five measurements are 10, 10, 10, 10, and 20.",
        "Is the median equal to 10?",
        1,
    ),
    (
        "A dataset contains 100 observations. 25 are missing completely from the recorded table.",
        "Does the recorded table contain 100 observed values?",
        0,
    ),
]

for i, (state, question, label) in enumerate(data_cases, 1):
    add(rows, f"E4CH_D{i:02d}", "data_reasoning", state, question, label)


# ============================================================
# VALIDATION
# ============================================================

assert len(rows) == 80, f"Expected 80 rows, got {len(rows)}"

labels = [r["label"] for r in rows]
assert labels.count(1) == 53, labels.count(1)
assert labels.count(0) == 27, labels.count(0)
domains = {}
for r in rows:
    domains[r["domain"]] = domains.get(r["domain"], 0) + 1

assert domains == {
    "mathematics": 20,
    "probability": 15,
    "formal_logic": 15,
    "science_reasoning": 15,
    "data_reasoning": 15,
}, domains

ids = [r["id"] for r in rows]
assert len(ids) == len(set(ids))

for r in rows:
    assert r["state"].strip()
    assert r["question"].strip()
    assert r["label"] in (0, 1)

# ============================================================
# WRITE
# ============================================================

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "domain",
            "state",
            "question",
            "label",
            "difficulty",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("E4c-H DATASET CREATED")
print(f"Rows: {len(rows)}")
print(f"True: {labels.count(1)}")
print(f"False: {labels.count(0)}")
print("Domains:")
for domain, count in domains.items():
    print(f"  {domain}: {count}")
print(f"Output: {OUT}")
print("VALIDATION PASSED")
