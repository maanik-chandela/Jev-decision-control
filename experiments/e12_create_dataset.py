import csv
from pathlib import Path


PROJECT_ROOT = Path.cwd()
OUT = PROJECT_ROOT / "experiments/data/e12_temporal_stability_dataset.csv"


CASES = [
    # =========================
    # MATHEMATICS
    # =========================
    ("M01", "mathematics", "easy", 1,
     "Is 7 + 5 equal to 12?",
     "Does the equation 7 + 5 = 12 hold?",
     "Consider the arithmetic statement: 7 plus 5 equals 12. Is it true?"),

    ("M02", "mathematics", "easy", 0,
     "Is 9 × 6 equal to 52?",
     "Does 9 multiplied by 6 give 52?",
     "Consider the arithmetic statement that 9 times 6 equals 52. Is it true?"),

    ("M03", "mathematics", "easy", 1,
     "Is the square of 8 equal to 64?",
     "Does 8 squared equal 64?",
     "Consider the mathematical statement: the square of 8 is 64. Is it true?"),

    ("M04", "mathematics", "easy", 0,
     "Is 15 divided by 3 equal to 4?",
     "Does 15 ÷ 3 equal 4?",
     "Consider the arithmetic claim that dividing 15 by 3 gives 4. Is it true?"),

    ("M05", "mathematics", "moderate", 1,
     "If a rectangle has length 8 and width 5, is its area 40?",
     "Does a rectangle measuring 8 by 5 have area 40?",
     "A rectangle has sides of length 8 and 5. Is its area 40 square units?"),

    ("M06", "mathematics", "moderate", 0,
     "If a square has side length 6, is its perimeter 20?",
     "Does a square with side 6 have perimeter 20?",
     "A square has side length 6 units. Is its perimeter 20 units?"),

    ("M07", "mathematics", "moderate", 1,
     "Is the average of 4, 8, and 12 equal to 8?",
     "Does the arithmetic mean of 4, 8, and 12 equal 8?",
     "The numbers are 4, 8, and 12. Is their mean 8?"),

    ("M08", "mathematics", "moderate", 0,
     "Is the average of 3, 9, and 12 equal to 9?",
     "Does the arithmetic mean of 3, 9, and 12 equal 9?",
     "The numbers are 3, 9, and 12. Is their mean 9?"),

    ("M09", "mathematics", "hard", 1,
     "If x + 7 = 15, must x equal 8?",
     "Does the equation x + 7 = 15 imply x = 8?",
     "Suppose x + 7 = 15. Is x necessarily 8?"),

    ("M10", "mathematics", "hard", 0,
     "If x² = 25, must x equal 5?",
     "Does x squared equal to 25 necessarily imply x = 5?",
     "Suppose x² = 25. Must x be exactly 5?"),

    ("M11", "mathematics", "hard", 1,
     "If two angles of a triangle are 50° and 60°, is the third angle 70°?",
     "Does a triangle with angles 50° and 60° have a third angle of 70°?",
     "A triangle has two angles measuring 50 and 60 degrees. Is its remaining angle 70 degrees?"),

    ("M12", "mathematics", "hard", 0,
     "If two angles of a triangle are 40° and 50°, is the third angle 100°?",
     "Does a triangle with angles 40° and 50° have a third angle of 100°?",
     "A triangle has two angles measuring 40 and 50 degrees. Is its remaining angle 100 degrees?"),

    # =========================
    # PROBABILITY
    # =========================
    ("P01", "probability", "easy", 1,
     "For a fair six-sided die, is the probability of rolling a 3 equal to 1/6?",
     "Is P(rolling a 3) = 1/6 for a fair six-sided die?",
     "A fair six-sided die is rolled. Is the probability of obtaining 3 equal to one sixth?"),

    ("P02", "probability", "easy", 0,
     "For a fair six-sided die, is the probability of rolling a 7 equal to 1/6?",
     "Is P(rolling a 7) = 1/6 for a fair six-sided die?",
     "A fair six-sided die is rolled. Is the probability of obtaining 7 equal to one sixth?"),

    ("P03", "probability", "easy", 1,
     "Is the probability of an impossible event equal to 0?",
     "Does an impossible event have probability 0?",
     "For an event that cannot occur, is its probability zero?"),

    ("P04", "probability", "easy", 0,
     "Is the probability of any event allowed to be greater than 1?",
     "Can a valid event probability exceed 1?",
     "In ordinary probability theory, can an event have probability greater than one?"),

    ("P05", "probability", "moderate", 1,
     "For a fair coin, is the probability of heads 1/2?",
     "Does a fair coin give probability 0.5 to heads?",
     "A coin is fair. Is the probability of heads one half?"),

    ("P06", "probability", "moderate", 0,
     "For a fair coin, is the probability of heads 2/3?",
     "Does a fair coin give probability two thirds to heads?",
     "A coin is fair. Is the probability of heads two thirds?"),

    ("P07", "probability", "moderate", 1,
     "If an event has probability 0.3, is its complement probability 0.7?",
     "Does an event with probability 0.3 have complement probability 0.7?",
     "An event has probability 0.3. Is the probability that it does not occur 0.7?"),

    ("P08", "probability", "moderate", 0,
     "If an event has probability 0.4, is its complement probability 0.4?",
     "Does an event with probability 0.4 have complement probability 0.4?",
     "An event has probability 0.4. Is the probability that it does not occur also 0.4?"),

    ("P09", "probability", "hard", 1,
    "If two independent events have probabilities 0.5 and 0.4, is the probability of their intersection 0.2?",
    "Two independent events have probabilities 0.5 and 0.4. Is their joint probability 0.2?",
    "Suppose events A and B are independent with P(A)=0.5 and P(B)=0.4. Is P(A and B)=0.2?"),

    ("P10", "probability", "hard", 0,
     "If two events are independent, must they be mutually exclusive?",
     "Does independence necessarily imply mutual exclusivity?",
     "Two events are independent. Must they therefore be mutually exclusive?"),

    ("P11", "probability", "hard", 1,
     "If an event has probability 1 in a finite discrete sample space, must it occur?",
     "In a finite discrete sample space, does probability 1 imply that an event occurs?",
     "Consider a finite discrete sample space. If an event has probability one, must the event occur?"),

    ("P12", "probability", "hard", 0,
     "If two events are mutually exclusive, must they be independent?",
     "Does mutual exclusivity necessarily imply independence?",
     "Two events are mutually exclusive and both have positive probability. Must they be independent?"),

    # =========================
    # FORMAL LOGIC
    # =========================
    ("L01", "formal_logic", "easy", 1,
     "If all cats are animals and Milo is a cat, must Milo be an animal?",
     "Does the premise that all cats are animals and Milo is a cat imply that Milo is an animal?",
     "All cats are animals. Milo is a cat. Must Milo therefore be an animal?"),

    ("L02", "formal_logic", "easy", 0,
     "If all cats are animals and Milo is an animal, must Milo be a cat?",
     "Does being an animal necessarily imply that Milo is a cat?",
     "All cats are animals. Milo is an animal. Must Milo therefore be a cat?"),

    ("L03", "formal_logic", "easy", 1,
    "If all birds are animals and all eagles are birds, must every eagle be an animal?",
    "Given that all birds are animals and all eagles are birds, does every eagle have to be an animal?",
    "All birds are animals. Every eagle is a bird. Must every eagle therefore be an animal?"),

    ("L04", "formal_logic", "easy", 0,
     "If some students are athletes, must every student be an athlete?",
     "Does the existence of student-athletes imply that all students are athletes?",
     "Some students are athletes. Must every student therefore be an athlete?"),

    ("L05", "formal_logic", "moderate", 1,
     "If A implies B and A is true, must B be true?",
     "Does A → B together with A imply B?",
     "Suppose A implies B and A is known to be true. Must B also be true?"),

    ("L06", "formal_logic", "moderate", 0,
     "If A implies B and B is true, must A be true?",
     "Does B being true together with A → B imply A?",
     "Suppose A implies B and B is known to be true. Must A also be true?"),

    ("L07", "formal_logic", "moderate", 1,
     "If A and B are both true, must A be true?",
     "Does A ∧ B imply A?",
     "Suppose both A and B are true. Must A itself be true?"),

    ("L08", "formal_logic", "moderate", 0,
     "If A or B is true, must both A and B be true?",
     "Does A ∨ B imply A ∧ B?",
     "If at least one of A and B is true, must both be true?"),

    ("L09", "formal_logic", "hard", 1,
    "If A implies B and B implies C, must A imply C?",
    "Does A → B together with B → C imply A → C?",
    "Suppose A implies B and B implies C. Must A therefore imply C?"),

    ("L10", "formal_logic", "hard", 0,
     "If some A are B and no B are C, must every A be outside C?",
     "Does some A being B together with no B being C imply that every A is not C?",
     "Some A are B, and no B is C. Must every A necessarily be non-C?"),

    ("L11", "formal_logic", "hard", 1,
     "If A implies B and B is false, must A be false?",
     "Does A → B together with not-B imply not-A?",
     "Suppose A implies B, but B is false. Must A therefore be false?"),

    ("L12", "formal_logic", "hard", 0,
     "If A implies B and A is false, must B be false?",
     "Does not-A together with A → B imply not-B?",
     "Suppose A implies B, but A is false. Must B therefore be false?"),

    # =========================
    # SCIENCE REASONING
    # =========================
    ("S01", "science_reasoning", "easy", 1,
     "Does water freeze at 0°C under standard atmospheric pressure?",
     "Under standard atmospheric pressure, does water freeze at 0°C?",
     "At standard atmospheric pressure, is 0 degrees Celsius the freezing point of water?"),

    ("S02", "science_reasoning", "easy", 0,
     "Does water boil at 50°C under standard atmospheric pressure?",
     "Under standard atmospheric pressure, does water boil at 50°C?",
     "At standard atmospheric pressure, is 50 degrees Celsius the boiling point of water?"),

    ("S03", "science_reasoning", "easy", 1,
     "Does the Earth orbit the Sun?",
     "Is the Earth in orbit around the Sun?",
     "Is Earth's motion around the Sun an orbital motion?"),

    ("S04", "science_reasoning", "easy", 0,
     "Does the Moon produce most of its own visible light?",
     "Is most of the Moon's visible light produced by the Moon itself?",
     "Does the Moon generate most of the visible light we observe from it?"),

    ("S05", "science_reasoning", "moderate", 1,
     "Does increasing the mass of an object while keeping acceleration fixed increase its net force?",
     "With acceleration fixed, does greater mass imply greater net force?",
     "If acceleration stays constant, does increasing mass increase the net force?"),

    ("S06", "science_reasoning", "moderate", 0,
     "If net force on an object is zero, must the object be stationary?",
     "Does zero net force necessarily mean an object is at rest?",
     "If the net force is zero, must the object's velocity be zero?"),

    ("S07", "science_reasoning", "moderate", 1,
     "Can plants perform photosynthesis using light energy?",
     "Does photosynthesis use light energy in plants?",
     "Is light energy used by plants during photosynthesis?"),

    ("S08", "science_reasoning", "moderate", 0,
     "Does photosynthesis occur only at night?",
     "Is photosynthesis restricted to nighttime?",
     "Do plants perform photosynthesis exclusively during the night?"),

    ("S09", "science_reasoning", "hard", 1,
     "If an object moves at constant velocity, is its acceleration zero?",
     "Does constant velocity imply zero acceleration?",
     "An object has constant velocity. Must its acceleration be zero?"),

    ("S10", "science_reasoning", "hard", 0,
     "If an object has zero acceleration, must its velocity be zero?",
     "Does zero acceleration necessarily imply zero velocity?",
     "An object has zero acceleration. Must it therefore have zero velocity?"),

    ("S11", "science_reasoning", "hard", 0,
     "Can sound propagate through a vacuum?",
     "Is sound able to travel through a vacuum?",
     "Can a sound wave propagate when there is no material medium?"),

    ("S12", "science_reasoning", "hard", 1,
     "Can light propagate through a vacuum?",
     "Is light able to travel through a vacuum?",
     "Can electromagnetic light propagate when there is no material medium?"),

    # =========================
    # DATA REASONING
    # =========================
    ("D01", "data_reasoning", "easy", 1,
     "If a dataset contains 10 values and 3 are missing, are 7 values observed?",
     "Does a dataset with 10 values and 3 missing values contain 7 observed values?",
     "A dataset has 10 entries, of which 3 are missing. Are 7 entries observed?"),

    ("D02", "data_reasoning", "easy", 0,
     "If 20 out of 100 observations are positive, is the positive proportion 30%?",
     "Is the positive proportion 30% when 20 of 100 observations are positive?",
     "A dataset contains 100 observations, 20 of which are positive. Is the positive proportion 30 percent?"),

    ("D03", "data_reasoning", "easy", 1,
     "If 80 out of 100 predictions are correct, is accuracy 80%?",
     "Does 80 correct predictions out of 100 correspond to 80% accuracy?",
     "A model makes 100 predictions and gets 80 correct. Is its accuracy 80 percent?"),

    ("D04", "data_reasoning", "easy", 0,
     "If 80 out of 100 predictions are correct, is accuracy 90%?",
     "Does 80 correct predictions out of 100 correspond to 90% accuracy?",
     "A model makes 100 predictions and gets 80 correct. Is its accuracy 90 percent?"),

    ("D05", "data_reasoning", "moderate", 1,
     "If a dataset has values 2, 4, 6, and 8, is its mean 5?",
     "Does the dataset {2,4,6,8} have mean 5?",
     "The values are 2, 4, 6, and 8. Is their arithmetic mean 5?"),

    ("D06", "data_reasoning", "moderate", 0,
     "If a dataset has values 2, 4, 6, and 8, is its median 6?",
     "Does the dataset {2,4,6,8} have median 6?",
     "The values are 2, 4, 6, and 8. Is their median 6?"),

    ("D07", "data_reasoning", "moderate", 1,
     "If 30 of 50 observations belong to class A, is class A the majority class?",
     "Does class A form the majority when it contains 30 of 50 observations?",
     "In a dataset of 50 observations, 30 are class A. Is class A the majority?"),

    ("D08", "data_reasoning", "moderate", 0,
     "If 20 of 50 observations belong to class A, is class A the majority class?",
     "Does class A form the majority when it contains 20 of 50 observations?",
     "In a dataset of 50 observations, 20 are class A. Is class A the majority?"),

    ("D09", "data_reasoning", "hard", 1,
     "If precision is 80%, does that mean 80% of predicted positives are actually positive?",
     "Does precision of 80% mean that 80% of predicted positives are correct?",
     "A classifier has precision 80 percent. Does this mean four out of five predicted positives are actually positive?"),

    ("D10", "data_reasoning", "hard", 0,
     "If recall is 80%, does that mean 80% of predicted positives are actually positive?",
     "Does recall of 80% mean that 80% of predicted positives are correct?",
     "A classifier has recall 80 percent. Does this mean four out of five predicted positives are actually positive?"),

    ("D11", "data_reasoning", "hard", 1,
     "If every value in a dataset is increased by 5, does the mean also increase by 5?",
     "Does adding 5 to every observation increase the mean by 5?",
     "Suppose 5 is added to every value in a dataset. Does the dataset mean increase by 5?"),

    ("D12", "data_reasoning", "hard", 0,
     "If every value in a dataset is multiplied by 2, does the mean remain unchanged?",
     "Does multiplying every observation by 2 leave the mean unchanged?",
     "Suppose every value in a dataset is doubled. Does the mean stay exactly the same?"),
]


def validate():
    assert len(CASES) == 60

    labels = [row[3] for row in CASES]
    assert labels.count(0) == 30
    assert labels.count(1) == 30

    domains = [row[1] for row in CASES]
    for domain in [
        "mathematics",
        "probability",
        "formal_logic",
        "science_reasoning",
        "data_reasoning",
    ]:
        assert domains.count(domain) == 12

    difficulties = [row[2] for row in CASES]
    assert difficulties.count("easy") == 20
    assert difficulties.count("moderate") == 20
    assert difficulties.count("hard") == 20

    ids = [row[0] for row in CASES]
    assert len(ids) == len(set(ids))

    for case_id, domain, difficulty, label, q1, q2, q3 in CASES:
        assert q1 != q2
        assert q2 != q3
        assert q1 != q3
        assert label in (0, 1)

    print("Validation PASSED.")


def write_dataset():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "domain",
        "difficulty",
        "label",
        "t1_question",
        "t2_question",
        "t3_question",
    ]

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in CASES:
            writer.writerow(
                {
                    "case_id": row[0],
                    "domain": row[1],
                    "difficulty": row[2],
                    "label": row[3],
                    "t1_question": row[4],
                    "t2_question": row[5],
                    "t3_question": row[6],
                }
            )

    print(f"Dataset written to: {OUT}")
    print(f"Total cases: {len(CASES)}")
    print("Labels:")
    print("  False:", sum(r[3] == 0 for r in CASES))
    print("  True :", sum(r[3] == 1 for r in CASES))


if __name__ == "__main__":
    validate()
    write_dataset()
