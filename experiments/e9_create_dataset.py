import pandas as pd
from pathlib import Path

OUTPUT = Path("experiments/data/e9_state_manipulation_dataset.csv")

cases = [

    # ========================================================
    # MATHEMATICS
    # ========================================================

    {
        "case_id": "M01",
        "domain": "mathematics",
        "label": 1,
        "state_original": "A rectangle has length 8 cm and width 5 cm.",
        "question": "Is its area 40 square centimeters?",
        "irrelevant_context": "A rectangle has length 8 cm and width 5 cm. The calculation is being done on a Monday.",
        "distracting_context": "A rectangle has length 8 cm and width 5 cm. A nearby square has side length 10 cm, and someone writes that its area is 100 square centimeters.",
    },

    {
        "case_id": "M02",
        "domain": "mathematics",
        "label": 0,
        "state_original": "A square has side length 6 cm.",
        "question": "Is its area 30 square centimeters?",
        "irrelevant_context": "A square has side length 6 cm. The square is drawn with a black pencil.",
        "distracting_context": "A square has side length 6 cm. A nearby rectangle has area 30 square centimeters.",
    },

    {
        "case_id": "M03",
        "domain": "mathematics",
        "label": 1,
        "state_original": "The number 18 is divisible by 3.",
        "question": "Is 18 divisible by 3?",
        "irrelevant_context": "The number 18 is divisible by 3. The number is printed in bold type.",
        "distracting_context": "The number 18 is divisible by 3. Another number, 20, is mentioned as an example of an even number.",
    },

    {
        "case_id": "M04",
        "domain": "mathematics",
        "label": 0,
        "state_original": "A triangle has angles measuring 60, 60, and 60 degrees.",
        "question": "Is one of its angles 90 degrees?",
        "irrelevant_context": "A triangle has angles measuring 60, 60, and 60 degrees. The diagram is shown on white paper.",
        "distracting_context": "A triangle has angles measuring 60, 60, and 60 degrees. A different right triangle is also mentioned.",
    },

    {
        "case_id": "M05",
        "domain": "mathematics",
        "label": 1,
        "state_original": "If x = 7, then x + 5 = 12.",
        "question": "Is x + 5 equal to 12?",
        "irrelevant_context": "If x = 7, then x + 5 = 12. The equation is written in a notebook.",
        "distracting_context": "If x = 7, then x + 5 = 12. Another example states that 7 + 6 = 13.",
    },

    {
        "case_id": "M06",
        "domain": "mathematics",
        "label": 0,
        "state_original": "A circle has radius 4 cm.",
        "question": "Is its diameter 6 cm?",
        "irrelevant_context": "A circle has radius 4 cm. The circle is colored gray.",
        "distracting_context": "A circle has radius 4 cm. A different circle has radius 3 cm and diameter 6 cm.",
    },

    {
        "case_id": "M07",
        "domain": "mathematics",
        "label": 1,
        "state_original": "The sequence is 2, 4, 6, 8, 10.",
        "question": "Is the next number 12?",
        "irrelevant_context": "The sequence is 2, 4, 6, 8, 10. The sequence appears in a textbook.",
        "distracting_context": "The sequence is 2, 4, 6, 8, 10. Another sequence is 1, 3, 5, 7.",
    },

    {
        "case_id": "M08",
        "domain": "mathematics",
        "label": 0,
        "state_original": "The integer 15 is less than 10.",
        "question": "Is 15 less than 10?",
        "irrelevant_context": "The integer 15 is less than 10. The statement is written in blue ink.",
        "distracting_context": "The integer 15 is less than 10. Another statement says that 5 is less than 10.",
    },

    {
        "case_id": "M09",
        "domain": "mathematics",
        "label": 1,
        "state_original": "A line segment has endpoints at 0 and 10 on a number line.",
        "question": "Is its length 10 units?",
        "irrelevant_context": "A line segment has endpoints at 0 and 10 on a number line. The number line is horizontal.",
        "distracting_context": "A line segment has endpoints at 0 and 10 on a number line. Another segment has endpoints at 0 and 6.",
    },

    {
        "case_id": "M10",
        "domain": "mathematics",
        "label": 0,
        "state_original": "A cube has 6 faces.",
        "question": "Does the cube have 8 faces?",
        "irrelevant_context": "A cube has 6 faces. The cube is made of wood.",
        "distracting_context": "A cube has 6 faces. A different geometric object has 8 faces.",
    },

    {
        "case_id": "M11",
        "domain": "mathematics",
        "label": 1,
        "state_original": "The average of 4 and 8 is 6.",
        "question": "Is the average equal to 6?",
        "irrelevant_context": "The average of 4 and 8 is 6. The values are written on a whiteboard.",
        "distracting_context": "The average of 4 and 8 is 6. Another pair of numbers, 2 and 10, also has average 6.",
    },

    {
        "case_id": "M12",
        "domain": "mathematics",
        "label": 0,
        "state_original": "A rectangle has length 6 cm and width 3 cm.",
        "question": "Is its perimeter 20 cm?",
        "irrelevant_context": "A rectangle has length 6 cm and width 3 cm. The rectangle is drawn with a ruler.",
        "distracting_context": "A rectangle has length 6 cm and width 3 cm. A different rectangle has perimeter 20 cm.",
    },

    # ========================================================
    # PROBABILITY
    # ========================================================

    {
        "case_id": "P01",
        "domain": "probability",
        "label": 1,
        "state_original": "A fair six-sided die is rolled once.",
        "question": "Is the probability of rolling a 6 equal to 1/6?",
        "irrelevant_context": "A fair six-sided die is rolled once. The die is on a wooden table.",
        "distracting_context": "A fair six-sided die is rolled once. A different example says that the probability of rolling an even number is 1/2.",
    },

    {
        "case_id": "P02",
        "domain": "probability",
        "label": 0,
        "state_original": "A fair six-sided die is rolled once.",
        "question": "Is the probability of rolling a 6 equal to 1/2?",
        "irrelevant_context": "A fair six-sided die is rolled once. The experiment takes place indoors.",
        "distracting_context": "A fair six-sided die is rolled once. The probability of rolling an even number is 1/2.",
    },

    {
        "case_id": "P03",
        "domain": "probability",
        "label": 1,
        "state_original": "A bag contains 4 red balls and 6 blue balls.",
        "question": "Is the probability of drawing a red ball 0.4?",
        "irrelevant_context": "A bag contains 4 red balls and 6 blue balls. The bag is placed on a table.",
        "distracting_context": "A bag contains 4 red balls and 6 blue balls. Another bag contains 5 red and 5 blue balls.",
    },

    {
        "case_id": "P04",
        "domain": "probability",
        "label": 0,
        "state_original": "A bag contains 2 green balls and 8 yellow balls.",
        "question": "Is the probability of drawing a green ball 0.8?",
        "irrelevant_context": "A bag contains 2 green balls and 8 yellow balls. The balls are small.",
        "distracting_context": "A bag contains 2 green balls and 8 yellow balls. Another bag has 8 green and 2 yellow balls.",
    },

    {
        "case_id": "P05",
        "domain": "probability",
        "label": 1,
        "state_original": "A coin is fair and is tossed once.",
        "question": "Is the probability of heads 0.5?",
        "irrelevant_context": "A coin is fair and is tossed once. The coin is silver.",
        "distracting_context": "A coin is fair and is tossed once. A different coin has been observed to land heads three times in a row.",
    },

    {
        "case_id": "P06",
        "domain": "probability",
        "label": 0,
        "state_original": "A fair die is rolled once.",
        "question": "Is the probability of rolling a number greater than 4 equal to 1/2?",
        "irrelevant_context": "A fair die is rolled once. The die is red.",
        "distracting_context": "A fair die is rolled once. The probability of rolling an even number is 1/2.",
    },

    {
        "case_id": "P07",
        "domain": "probability",
        "label": 1,
        "state_original": "A class has 20 students, 5 of whom wear glasses.",
        "question": "Is the proportion of students wearing glasses 0.25?",
        "irrelevant_context": "A class has 20 students, 5 of whom wear glasses. The class meets in Room 12.",
        "distracting_context": "A class has 20 students, 5 of whom wear glasses. Another class has 10 students, 5 of whom wear glasses.",
    },

    {
        "case_id": "P08",
        "domain": "probability",
        "label": 0,
        "state_original": "A box contains 10 cards, 3 of which are red.",
        "question": "Is the probability of selecting a red card 0.7?",
        "irrelevant_context": "A box contains 10 cards, 3 of which are red. The box is made of cardboard.",
        "distracting_context": "A box contains 10 cards, 3 of which are red. Another box contains 7 red cards out of 10.",
    },

    {
        "case_id": "P09",
        "domain": "probability",
        "label": 1,
        "state_original": "In a finite sample space where every elementary outcome has positive probability, an event has probability 0.",
        "question": "Is the event impossible?",
        "irrelevant_context": "In a finite sample space where every elementary outcome has positive probability, an event has probability 0. The sample space is written formally.",
        "distracting_context": "In a finite sample space where every elementary outcome has positive probability, an event has probability 0. Another example discusses a very unlikely event.",
    },

    {
        "case_id": "P10",
        "domain": "probability",
        "label": 0,
        "state_original": "A fair coin is tossed twice.",
        "question": "Is the probability of getting two heads equal to 1/2?",
        "irrelevant_context": "A fair coin is tossed twice. The coin is made of metal.",
        "distracting_context": "A fair coin is tossed twice. The probability of getting at least one head is 3/4.",
    },

    {
        "case_id": "P11",
        "domain": "probability",
        "label": 1,
        "state_original": "A bag contains 5 red balls and 5 blue balls.",
        "question": "Is the probability of drawing a red ball 0.5?",
        "irrelevant_context": "A bag contains 5 red balls and 5 blue balls. The balls are numbered.",
        "distracting_context": "A bag contains 5 red balls and 5 blue balls. Another bag contains 8 red balls and 2 blue balls.",
    },

    {
        "case_id": "P12",
        "domain": "probability",
        "label": 0,
        "state_original": "A fair six-sided die is rolled once.",
        "question": "Is the probability of rolling a number less than 3 equal to 2/3?",
        "irrelevant_context": "A fair six-sided die is rolled once. The die is plastic.",
        "distracting_context": "A fair six-sided die is rolled once. The probability of rolling an even number is 1/2.",
    },

    # ========================================================
    # FORMAL LOGIC
    # ========================================================

    {
        "case_id": "L01",
        "domain": "formal_logic",
        "label": 1,
        "state_original": "All cats are mammals. Luna is a cat.",
        "question": "Must Luna be a mammal?",
        "irrelevant_context": "All cats are mammals. Luna is a cat. Luna lives indoors.",
        "distracting_context": "All cats are mammals. Luna is a cat. Dogs are also mammals, but Bruno is a dog.",
    },

    {
        "case_id": "L02",
        "domain": "formal_logic",
        "label": 0,
        "state_original": "All cats are mammals. Luna is a cat.",
        "question": "Must Luna be a reptile?",
        "irrelevant_context": "All cats are mammals. Luna is a cat. Luna has a collar.",
        "distracting_context": "All cats are mammals. Luna is a cat. Some reptiles are green.",
    },

    {
        "case_id": "L03",
        "domain": "formal_logic",
        "label": 1,
        "state_original": "If A occurs, then B occurs. A occurs.",
        "question": "Must B occur?",
        "irrelevant_context": "If A occurs, then B occurs. A occurs. The statement is written in a notebook.",
        "distracting_context": "If A occurs, then B occurs. A occurs. Another statement says that C implies D.",
    },

    {
        "case_id": "L04",
        "domain": "formal_logic",
        "label": 0,
        "state_original": "If A occurs, then B occurs. B occurs.",
        "question": "Must A have occurred?",
        "irrelevant_context": "If A occurs, then B occurs. B occurs. The symbols are written in black ink.",
        "distracting_context": "If A occurs, then B occurs. B occurs. Another implication states that C implies D.",
    },

    {
        "case_id": "L05",
        "domain": "formal_logic",
        "label": 1,
        "state_original": "No birds are mammals. Eagles are birds.",
        "question": "Must an eagle not be a mammal?",
        "irrelevant_context": "No birds are mammals. Eagles are birds. The eagle is shown in a diagram.",
        "distracting_context": "No birds are mammals. Eagles are birds. Whales are mammals.",
    },

    {
        "case_id": "L06",
        "domain": "formal_logic",
        "label": 0,
        "state_original": "No birds are mammals. Eagles are birds.",
        "question": "Must an eagle be a mammal?",
        "irrelevant_context": "No birds are mammals. Eagles are birds. The statement is printed in a textbook.",
        "distracting_context": "No birds are mammals. Eagles are birds. Some whales are mammals.",
    },

    {
        "case_id": "L07",
        "domain": "formal_logic",
        "label": 1,
        "state_original": "All engineers are graduates. Ravi is an engineer.",
        "question": "Must Ravi be a graduate?",
        "irrelevant_context": "All engineers are graduates. Ravi is an engineer. Ravi wears glasses.",
        "distracting_context": "All engineers are graduates. Ravi is an engineer. Some graduates are musicians.",
    },

    {
        "case_id": "L08",
        "domain": "formal_logic",
        "label": 0,
        "state_original": "All engineers are graduates. Ravi is an engineer.",
        "question": "Must Ravi be a professor?",
        "irrelevant_context": "All engineers are graduates. Ravi is an engineer. Ravi lives in Delhi.",
        "distracting_context": "All engineers are graduates. Ravi is an engineer. Some professors are graduates.",
    },

    {
        "case_id": "L09",
        "domain": "formal_logic",
        "label": 1,
        "state_original": "No dogs are cats. Bruno is a dog.",
        "question": "Must Bruno not be a cat?",
        "irrelevant_context": "No dogs are cats. Bruno is a dog. Bruno has a red collar.",
        "distracting_context": "No dogs are cats. Bruno is a dog. Some cats are black.",
    },

    {
        "case_id": "L10",
        "domain": "formal_logic",
        "label": 0,
        "state_original": "Some students are athletes. Ravi is a student.",
        "question": "Must Ravi be an athlete?",
        "irrelevant_context": "Some students are athletes. Ravi is a student. Ravi studies mathematics.",
        "distracting_context": "Some students are athletes. Ravi is a student. Another student, Arjun, is an athlete.",
    },

    {
        "case_id": "L11",
        "domain": "formal_logic",
        "label": 1,
        "state_original": "If P then Q. P is true.",
        "question": "Must Q be true?",
        "irrelevant_context": "If P then Q. P is true. The symbols are written on a whiteboard.",
        "distracting_context": "If P then Q. P is true. Another rule states that R implies S.",
    },

    {
        "case_id": "L12",
        "domain": "formal_logic",
        "label": 0,
        "state_original": "If P then Q. Q is true.",
        "question": "Must P be true?",
        "irrelevant_context": "If P then Q. Q is true. The statement appears in a logic exercise.",
        "distracting_context": "If P then Q. Q is true. Another rule states that R implies S.",
    },

    # ========================================================
    # SCIENCE REASONING
    # ========================================================

    {
        "case_id": "S01",
        "domain": "science_reasoning",
        "label": 1,
        "state_original": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "question": "Does water freeze at 0 degrees Celsius under these conditions?",
        "irrelevant_context": "Water freezes at 0 degrees Celsius under standard atmospheric pressure. The container is made of glass.",
        "distracting_context": "Water freezes at 0 degrees Celsius under standard atmospheric pressure. Ethanol freezes at a lower temperature.",
    },

    {
        "case_id": "S02",
        "domain": "science_reasoning",
        "label": 0,
        "state_original": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "question": "Does water freeze at 100 degrees Celsius under these conditions?",
        "irrelevant_context": "Water freezes at 0 degrees Celsius under standard atmospheric pressure. The water is clear.",
        "distracting_context": "Water freezes at 0 degrees Celsius under standard atmospheric pressure. Water boils at approximately 100 degrees Celsius.",
    },

    {
        "case_id": "S03",
        "domain": "science_reasoning",
        "label": 1,
        "state_original": "Plants use sunlight during photosynthesis.",
        "question": "Does photosynthesis use sunlight?",
        "irrelevant_context": "Plants use sunlight during photosynthesis. The plant is in a ceramic pot.",
        "distracting_context": "Plants use sunlight during photosynthesis. Plants also require water.",
    },

    {
        "case_id": "S04",
        "domain": "science_reasoning",
        "label": 0,
        "state_original": "Plants use sunlight during photosynthesis.",
        "question": "Does photosynthesis require no energy input?",
        "irrelevant_context": "Plants use sunlight during photosynthesis. The plant has green leaves.",
        "distracting_context": "Plants use sunlight during photosynthesis. Cellular respiration releases energy.",
    },

    {
        "case_id": "S05",
        "domain": "science_reasoning",
        "label": 1,
        "state_original": "The Earth revolves around the Sun.",
        "question": "Does the Earth revolve around the Sun?",
        "irrelevant_context": "The Earth revolves around the Sun. The statement is printed in a textbook.",
        "distracting_context": "The Earth revolves around the Sun. The Moon revolves around the Earth.",
    },

    {
        "case_id": "S06",
        "domain": "science_reasoning",
        "label": 0,
        "state_original": "The Earth revolves around the Sun.",
        "question": "Does the Sun revolve around the Earth?",
        "irrelevant_context": "The Earth revolves around the Sun. The diagram is drawn on a whiteboard.",
        "distracting_context": "The Earth revolves around the Sun. The Moon revolves around the Earth.",
    },

    {
        "case_id": "S07",
        "domain": "science_reasoning",
        "label": 1,
        "state_original": "Sound requires a medium to propagate.",
        "question": "Can sound propagate through a material medium?",
        "irrelevant_context": "Sound requires a medium to propagate. The experiment is performed indoors.",
        "distracting_context": "Sound requires a medium to propagate. Light can travel through vacuum.",
    },

    {
        "case_id": "S08",
        "domain": "science_reasoning",
        "label": 0,
        "state_original": "Sound requires a medium to propagate.",
        "question": "Can ordinary sound propagate through a perfect vacuum?",
        "irrelevant_context": "Sound requires a medium to propagate. The statement is written in a notebook.",
        "distracting_context": "Sound requires a medium to propagate. Electromagnetic waves can travel through vacuum.",
    },

    {
        "case_id": "S09",
        "domain": "science_reasoning",
        "label": 1,
        "state_original": "An object with greater mass has greater inertia, all else equal.",
        "question": "Does greater mass imply greater inertia?",
        "irrelevant_context": "An object with greater mass has greater inertia, all else equal. The objects are measured in kilograms.",
        "distracting_context": "An object with greater mass has greater inertia, all else equal. A different experiment compares object speeds.",
    },

    {
        "case_id": "S10",
        "domain": "science_reasoning",
        "label": 0,
        "state_original": "An object with greater mass has greater inertia, all else equal.",
        "question": "Does greater mass imply zero inertia?",
        "irrelevant_context": "An object with greater mass has greater inertia, all else equal. The statement is in a physics textbook.",
        "distracting_context": "An object with greater mass has greater inertia, all else equal. Another object has a different velocity.",
    },

    {
        "case_id": "S11",
        "domain": "science_reasoning",
        "label": 1,
        "state_original": "Metals generally conduct electricity.",
        "question": "Can metals conduct electricity?",
        "irrelevant_context": "Metals generally conduct electricity. The sample is placed on a laboratory table.",
        "distracting_context": "Metals generally conduct electricity. Rubber is an electrical insulator.",
    },

    {
        "case_id": "S12",
        "domain": "science_reasoning",
        "label": 0,
        "state_original": "Metals generally conduct electricity.",
        "question": "Are metals generally perfect electrical insulators?",
        "irrelevant_context": "Metals generally conduct electricity. The sample is silver-colored.",
        "distracting_context": "Metals generally conduct electricity. Rubber is commonly used as an insulator.", 
    },

    # ========================================================
    # DATA REASONING
    # ========================================================

    {
        "case_id": "D01",
        "domain": "data_reasoning",
        "label": 1,
        "state_original": "A dataset contains values 2, 4, 6, 8, 10.",
        "question": "Is the mean equal to 6?",
        "irrelevant_context": "A dataset contains values 2, 4, 6, 8, 10. The values are stored in a spreadsheet.",
        "distracting_context": "A dataset contains values 2, 4, 6, 8, 10. Another dataset has mean 20.",
    },

    {
        "case_id": "D02",
        "domain": "data_reasoning",
        "label": 0,
        "state_original": "A dataset contains values 2, 4, 6, 8, 10.",
        "question": "Is the mean equal to 8?",
        "irrelevant_context": "A dataset contains values 2, 4, 6, 8, 10. The data are arranged in ascending order.",
        "distracting_context": "A dataset contains values 2, 4, 6, 8, 10. Another dataset has mean 8.",
    },

    {
        "case_id": "D03",
        "domain": "data_reasoning",
        "label": 1,
        "state_original": "Out of 100 observations, 20 belong to category A.",
        "question": "Is the proportion of category A equal to 20%?",
        "irrelevant_context": "Out of 100 observations, 20 belong to category A. The observations are stored in a CSV file.",
        "distracting_context": "Out of 100 observations, 20 belong to category A. Another dataset has 40 observations in category A.",
    },

    {
        "case_id": "D04",
        "domain": "data_reasoning",
        "label": 0,
        "state_original": "Out of 100 observations, 20 belong to category A.",
        "question": "Is the proportion of category A equal to 80%?",
        "irrelevant_context": "Out of 100 observations, 20 belong to category A. The dataset is shown as a table.",
        "distracting_context": "Out of 100 observations, 20 belong to category A. Category B contains 80 observations.",
    },

    {
        "case_id": "D05",
        "domain": "data_reasoning",
        "label": 1,
        "state_original": "A survey reports 80 positive responses out of 100 respondents.",
        "question": "Is the positive response rate 80%?",
        "irrelevant_context": "A survey reports 80 positive responses out of 100 respondents. The survey was conducted online.",
        "distracting_context": "A survey reports 80 positive responses out of 100 respondents. Another survey reports 60 positive responses out of 100.",
    },

    {
        "case_id": "D06",
        "domain": "data_reasoning",
        "label": 0,
        "state_original": "A survey reports 80 positive responses out of 100 respondents.",
        "question": "Is the positive response rate 20%?",
        "irrelevant_context": "A survey reports 80 positive responses out of 100 respondents. The questionnaire had five questions.",
        "distracting_context": "A survey reports 80 positive responses out of 100 respondents. Another survey reports 20 positive responses out of 100.",
    },

    {
        "case_id": "D07",
        "domain": "data_reasoning",
        "label": 1,
        "state_original": "A dataset contains values 3, 5, 7, 9, 11.",
        "question": "Is the median equal to 7?",
        "irrelevant_context": "A dataset contains values 3, 5, 7, 9, 11. The values are sorted.",
        "distracting_context": "A dataset contains values 3, 5, 7, 9, 11. Another dataset has median 9.",
    },

    {
        "case_id": "D08",
        "domain": "data_reasoning",
        "label": 0,
        "state_original": "A dataset contains values 3, 5, 7, 9, 11.",
        "question": "Is the median equal to 9?",
        "irrelevant_context": "A dataset contains values 3, 5, 7, 9, 11. The data are numeric.",
        "distracting_context": "A dataset contains values 3, 5, 7, 9, 11. Another dataset has median 9.",
    },

    {
        "case_id": "D09",
        "domain": "data_reasoning",
        "label": 1,
        "state_original": "A classifier makes 90 correct predictions out of 100 cases.",
        "question": "Is its accuracy 90%?",
        "irrelevant_context": "A classifier makes 90 correct predictions out of 100 cases. The predictions are stored in a table.",
        "distracting_context": "A classifier makes 90 correct predictions out of 100 cases. Another classifier has 80 correct predictions out of 100.",
    },

    {
        "case_id": "D10",
        "domain": "data_reasoning",
        "label": 0,
        "state_original": "A classifier makes 90 correct predictions out of 100 cases.",
        "question": "Is its accuracy 10%?",
        "irrelevant_context": "A classifier makes 90 correct predictions out of 100 cases. The evaluation is performed automatically.",
        "distracting_context": "A classifier makes 90 correct predictions out of 100 cases. Its error rate is 10%.",
    },

    {
        "case_id": "D11",
        "domain": "data_reasoning",
        "label": 1,
        "state_original": "A table contains 50 rows and 4 columns.",
        "question": "Does the table contain 200 cells?",
        "irrelevant_context": "A table contains 50 rows and 4 columns. The table is displayed on a computer.",
        "distracting_context": "A table contains 50 rows and 4 columns. Another table has 100 rows and 2 columns.",
    },

    {
        "case_id": "D12",
        "domain": "data_reasoning",
        "label": 0,
        "state_original": "A table contains 50 rows and 4 columns.",
        "question": "Does the table contain 100 cells?",
        "irrelevant_context": "A table contains 50 rows and 4 columns. The table is saved as a spreadsheet.",
        "distracting_context": "A table contains 50 rows and 4 columns. Another table contains 100 cells.",
    },
]

rows = []

for case in cases:

    variants = [
        ("original", case["state_original"]),
        ("irrelevant_context", case["irrelevant_context"]),
        ("distracting_context", case["distracting_context"]),
    ]

    for condition, state in variants:

        rows.append({
            "case_id": case["case_id"],
            "domain": case["domain"],
            "condition": condition,
            "label": case["label"],
            "state": state,
            "question": case["question"],
        })

df = pd.DataFrame(rows)

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

assert len(df) == 180
assert df["case_id"].nunique() == 60
assert df.groupby("case_id").size().eq(3).all()
assert df.groupby("case_id")["label"].nunique().eq(1).all()

assert df["condition"].value_counts().to_dict() == {
    "original": 60,
    "irrelevant_context": 60,
    "distracting_context": 60,
}

assert df["label"].value_counts().to_dict() == {
    0: 90,
    1: 90,
}

assert df.groupby(["domain", "case_id"]).size().eq(3).all()

print("=" * 70)
print("E9 DATASET VALIDATION")
print("=" * 70)

print(f"Cases: {df['case_id'].nunique()}")
print(f"Rows: {len(df)}")
print(f"Labels:\n{df['label'].value_counts().sort_index()}")
print(f"\nConditions:\n{df['condition'].value_counts()}")

print("\nDomains:")
print(df.groupby(["domain", "condition"]).size())

df.to_csv(OUTPUT, index=False)

print(f"\nSaved dataset to: {OUTPUT}")
