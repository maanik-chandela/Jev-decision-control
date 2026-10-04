import pandas as pd
from pathlib import Path

# E9b: Controlled State Manipulation Robustness
#
# Purpose:
# Test whether JEV's probability changes when additional
# irrelevant or misleading information is added, while
# keeping the target proposition explicitly identifiable.
#
# Conditions:
#   original
#   irrelevant_context
#   distracting_context
#
# 60 underlying decisions x 3 conditions = 180 evaluations.

cases = [
    # -------------------------
    # MATHEMATICS
    # -------------------------
    {
        "case_id": "M01",
        "domain": "mathematics",
        "label": 1,
        "target_state": "A square has side length 6 cm.",
        "target_question": "Is the area of the square 36 square centimeters?",
        "irrelevant": "A triangle is also drawn nearby, but its dimensions are not relevant.",
        "distracting": "A different rectangle has length 5 cm and width 4 cm. The square is still the target object."
    },
    {
        "case_id": "M02",
        "domain": "mathematics",
        "label": 0,
        "target_state": "A square has side length 5 cm.",
        "target_question": "Is the area of the square 30 square centimeters?",
        "irrelevant": "A triangle has base 8 cm and height 3 cm.",
        "distracting": "A different rectangle has area 30 square centimeters. The square remains the target object."
    },
    {
        "case_id": "M03",
        "domain": "mathematics",
        "label": 1,
        "target_state": "A rectangle has length 8 cm and width 3 cm.",
        "target_question": "Is the area of the rectangle 24 square centimeters?",
        "irrelevant": "A circle with radius 2 cm is mentioned.",
        "distracting": "A different square has side length 5 cm. The rectangle remains the target object."
    },
    {
        "case_id": "M04",
        "domain": "mathematics",
        "label": 0,
        "target_state": "A rectangle has length 7 cm and width 2 cm.",
        "target_question": "Is the area of the first rectangle 20 square centimeters?",
        "irrelevant": "A triangle has three sides of lengths 3, 4, and 5 cm.",
        "distracting": "A different rectangle has area 20 square centimeters. The first rectangle remains the target object."
    },
    {
        "case_id": "M05",
        "domain": "mathematics",
        "label": 1,
        "target_state": "A circle has radius 4 cm.",
        "target_question": "Is the diameter of the first circle 8 cm?",
        "irrelevant": "A square with side length 10 cm is also described.",
        "distracting": "A different circle has radius 3 cm and diameter 6 cm. The first circle remains the target object."
    },
    {
        "case_id": "M06",
        "domain": "mathematics",
        "label": 0,
        "target_state": "A circle has radius 4 cm.",
        "target_question": "Is the diameter of the first circle 6 cm?",
        "irrelevant": "A triangle with base 5 cm and height 4 cm is also described.",
        "distracting": "A different circle has radius 3 cm and diameter 6 cm. The first circle remains the target object."
    },
    {
        "case_id": "M07",
        "domain": "mathematics",
        "label": 1,
        "target_state": "A triangle has base 10 cm and height 6 cm.",
        "target_question": "Is the area of the first triangle 30 square centimeters?",
        "irrelevant": "A rectangle with length 9 cm and width 2 cm is mentioned.",
        "distracting": "A different triangle has area 20 square centimeters. The first triangle remains the target object."
    },
    {
        "case_id": "M08",
        "domain": "mathematics",
        "label": 0,
        "target_state": "A triangle has base 10 cm and height 6 cm.",
        "target_question": "Is the area of the first triangle 40 square centimeters?",
        "irrelevant": "A circle with diameter 12 cm is mentioned.",
        "distracting": "A different triangle has area 40 square centimeters. The first triangle remains the target object."
    },
    {
        "case_id": "M09",
        "domain": "mathematics",
        "label": 1,
        "target_state": "An integer is 17.",
        "target_question": "Is the integer greater than 10?",
        "irrelevant": "Another integer is 4.",
        "distracting": "A different integer is 4, but the target integer is still 17."
    },
	{
    	"case_id": "M10",
    	"domain": "mathematics",
    	"label": 0,
    	"target_state": "An integer is 17.",
    	"target_question": "Is the integer less than 10?",
    	"irrelevant": "A triangle has three sides.",
    	"distracting": "A different integer is 4, but the target integer is still 17."
	},
 	{
        "case_id": "M11",
        "domain": "mathematics",
        "label": 1,
        "target_state": "The sum of 8 and 7 is 15.",
        "target_question": "Is the stated sum equal to 15?",
        "irrelevant": "The product of 3 and 4 is also mentioned.",
        "distracting": "A different calculation, 8 + 6, equals 14. The target calculation remains 8 + 7."
   	 },
    {
        "case_id": "M12",
        "domain": "mathematics",
        "label": 0,
        "target_state": "A rectangle has length 6 cm and width 3 cm.",
        "target_question": "Is the perimeter of the target rectangle 20 cm?",
        "irrelevant": "A circle has radius 2 cm.",
        "distracting": "A different rectangle has perimeter 18 cm. The target rectangle has dimensions 6 cm by 3 cm."
    },

    # -------------------------
    # PROBABILITY
    # -------------------------
    {
        "case_id": "P01",
        "domain": "probability",
        "label": 1,
        "target_state": "A fair six-sided die is rolled once.",
        "target_question": "Is the probability of rolling a 6 on the target die equal to 1/6?",
        "irrelevant": "A coin is also tossed once.",
        "distracting": "A different die is weighted toward 6. The target die is the fair die."
    },
    {
        "case_id": "P02",
        "domain": "probability",
        "label": 0,
        "target_state": "A fair six-sided die is rolled once.",
        "target_question": "Is the probability of rolling a 6 on the target die equal to 1/2?",
        "irrelevant": "A coin is also tossed once.",
        "distracting": "A different die has probability 1/2 of rolling a 6. The target die is fair."
    },
    {
        "case_id": "P03",
        "domain": "probability",
        "label": 1,
        "target_state": "A fair coin is tossed once.",
        "target_question": "Is the probability of heads on the target coin 0.5?",
        "irrelevant": "A six-sided die is also rolled.",
        "distracting": "A different biased coin has probability 0.8 of heads. The target coin is fair."
    },
    {
        "case_id": "P04",
        "domain": "probability",
        "label": 0,
        "target_state": "A fair coin is tossed once.",
        "target_question": "Is the probability of heads on the target coin 0.8?",
        "irrelevant": "A die is also rolled.",
        "distracting": "A different biased coin has probability 0.8 of heads. The target coin is fair."
    },
    {
        "case_id": "P05",
        "domain": "probability",
        "label": 1,
        "target_state": "A bag contains 3 red balls and 7 blue balls.",
        "target_question": "Is the probability of drawing a red ball from the target bag 0.3?",
        "irrelevant": "A second bag contains 5 green balls.",
        "distracting": "A different bag contains 8 red balls and 2 blue balls. The target bag contains 3 red and 7 blue balls."
    },
    {
        "case_id": "P06",
        "domain": "probability",
        "label": 0,
        "target_state": "A bag contains 3 red balls and 7 blue balls.",
        "target_question": "Is the probability of drawing a red ball from the taget bag 0.5?",
        "irrelevant": "A second bag contains 5 green balls.",
        "distracting": "A different bag contains 8 red balls and 2 blue balls. The target bag contains 3 red and 7 blue balls."
    },
    {
        "case_id": "P07",
        "domain": "probability",
        "label": 1,
        "target_state": "A fair die is rolled once.",
        "target_question": "Is the probability of rolling a number greater than 4 on the target die equal to 1/3?",
        "irrelevant": "A coin is tossed after the die roll.",
        "distracting": "A different die has a 1/2 probability of producing a number greater than 4. The target die is fair."
    },
    {
        "case_id": "P08",
        "domain": "probability",
        "label": 0,
        "target_state": "A box contains 3 red cards and 7 blue cards.",
        "target_question": "Is the probability of selecting a red card from the first box 0.7?",
        "irrelevant": "A second box contains 5 green cards.",
        "distracting": "A second box contains 7 red cards and 3 blue cards. The target is the first box, which contains 3 red and 7 blue cards."
    },
    {
        "case_id": "P09",
        "domain": "probability",
        "label": 1,
        "target_state": "In a finite sample space, an event has probability 0.",
        "target_question": "Is the target event impossible?",
        "irrelevant": "Another event has probability 0.5.",
        "distracting": "A different event is certain to occur. The target event has probability 0 in the finite sample space."
    },
    {
        "case_id": "P10",
        "domain": "probability",
        "label": 0,
        "target_state": "A fair six-sided die is rolled once.",
        "target_question": "Is the probability of rolling an even number equal to 1/3?",
        "irrelevant": "A coin is tossed separately.",
        "distracting": "A different experiment has probability 1/3 of producing an even result. The target die is fair."
    },
    {
        "case_id": "P11",
        "domain": "probability",
        "label": 1,
        "target_state": "A bag contains 8 red balls and 2 blue balls.",
        "target_question": "Is the probability of drawing a red ball from the first bag 0.8?",
        "irrelevant": "A second bag contains 5 red and 5 blue balls.",
        "distracting": "A second bag contains 5 red and 5 blue balls. The target is the first bag with 8 red and 2 blue balls."
    },
    {
        "case_id": "P12",
        "domain": "probability",
        "label": 0,
        "target_state": "A bag contains 8 red balls and 2 blue balls.",
        "target_question": "Is the probability of drawing a red ball from the first bag 0.5?",
        "irrelevant": "A second bag contains 5 red and 5 blue balls.",
        "distracting": "A second bag contains 5 red and 5 blue balls. The target is the first bag with 8 red and 2 blue balls."
    },

    # -------------------------
    # FORMAL LOGIC
    # -------------------------
    {
        "case_id": "L01",
        "domain": "formal_logic",
        "label": 1,
        "target_state": "If A occurs, then B occurs. A occurs.",
        "target_question": "Must B occur?",
        "irrelevant": "A separate statement says C occurs.",
        "distracting": "Another rule says C implies D. The target rule remains A implies B."
    },
    {
        "case_id": "L02",
        "domain": "formal_logic",
        "label": 0,
        "target_state": "If A occurs, then B occurs. A does not occur.",
        "target_question": "Must B not occur?",
        "irrelevant": "A separate statement says C occurs.",
        "distracting": "Another rule says C implies B. The target rule concerns A implying B."
    },
    {
        "case_id": "L03",
        "domain": "formal_logic",
        "label": 0,
        "target_state": "If A occurs, then B occurs. B occurs.",
        "target_question": "Must A have occurred?",
        "irrelevant": "C also occurs.",
        "distracting": "A separate rule states that C implies B. The target rule is still A implies B."
    },
    {
        "case_id": "L04",
        "domain": "formal_logic",
        "label": 1,
        "target_state": "If A occurs, then B occurs. B does not occur.",
        "target_question": "Must A not have occurred?",
        "irrelevant": "C is known to occur.",
        "distracting": "A separate rule states C implies D. The target rule is still A implies B."
    },
    {
        "case_id": "L05",
        "domain": "formal_logic",
        "label": 1,
        "target_state": "All A are B. X is A.",
        "target_question": "Must X be B?",
        "irrelevant": "Y is C.",
        "distracting": "All C are D and Y is C. The target statement remains all A are B."
    },
    {
        "case_id": "L06",
        "domain": "formal_logic",
        "label": 0,
        "target_state": "All A are B. X is not A.",
        "target_question": "Must X not be B?",
        "irrelevant": "Y is C.",
        "distracting": "All C are B and Y is C. The target statement remains all A are B."
    },
    {
        "case_id": "L07",
        "domain": "formal_logic",
        "label": 1,
        "target_state": "No A are B. X is A.",
        "target_question": "Must X not be B?",
        "irrelevant": "Y is C.",
        "distracting": "Some C are B. The target statement remains that no A are B."
    },
    {
        "case_id": "L08",
        "domain": "formal_logic",
        "label": 0,
        "target_state": "Some A are B.",
        "target_question": "Must every A be B?",
        "irrelevant": "Some C are D.",
        "distracting": "Every C is D. The target statement is only that some A are B."
    },
    {
        "case_id": "L09",
        "domain": "formal_logic",
        "label": 0,
        "target_state": "No A are B. All C are A.",
        "target_question": "Must every C be B?",
        "irrelevant": "Some D are E.",
        "distracting": "All D are B. The target relation remains C is a subset of A, with A disjoint from B."
    },
    {
        "case_id": "L10",
        "domain": "formal_logic",
        "label": 1,
        "target_state": "All birds in a specified group are animals. Penguin P is a bird in that group.",
        "target_question": "Must Penguin P be an animal?",
        "irrelevant": "A fish is also mentioned.",
        "distracting": "A different group contains a non-bird animal. Penguin P remains in the specified bird group."
    },
    {
        "case_id": "L11",
        "domain": "formal_logic",
        "label": 1,
        "target_state": "All birds in a specified group are animals. X is not an animal.",
        "target_question": "Must X not be a bird in that group?",
        "irrelevant": "A fish is mentioned.",
        "distracting": "A different bird group is described. The target group is the one in which all birds are animals."
    },
    {
        "case_id": "L12",
        "domain": "formal_logic",
        "label": 0,
        "target_state": "If a device is certified, then it passes inspection. Device X is certified.",
        "target_question": "Must Device X fail inspection?",
        "irrelevant": "Device Y has not yet been inspected.",
        "distracting": "A different device failed inspection. Device X remains certified under the target rule."
    },

    # -------------------------
    # SCIENCE REASONING
    # -------------------------
    {
        "case_id": "S01",
        "domain": "science_reasoning",
        "label": 1,
        "target_state": "Pure water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "target_question": "Does pure water freeze at 0 degrees Celsius under standard atmospheric pressure?",
        "irrelevant": "Iron melts at a much higher temperature.",
        "distracting": "Salt water can freeze below 0 degrees Celsius. The target substance is pure water."
    },
    {
        "case_id": "S02",
        "domain": "science_reasoning",
        "label": 0,
        "target_state": "Pure water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "target_question": "Does pure water freeze at 10 degrees Celsius under standard atmospheric pressure?",
        "irrelevant": "Copper conducts electricity.",
        "distracting": "A salt solution freezes below 0 degrees Celsius. The target substance is pure water."
    },
    {
        "case_id": "S03",
        "domain": "science_reasoning",
        "label": 1,
        "target_state": "Objects near Earth's surface accelerate downward at approximately 9.8 m/s^2 when air resistance is neglected.",
        "target_question": "Is the acceleration approximately 9.8 m/s^2?",
        "irrelevant": "The object also has a mass of 2 kg.",
        "distracting": "A different planet has a different gravitational acceleration. The target is near Earth's surface."
    },
    {
        "case_id": "S04",
        "domain": "science_reasoning",
        "label": 0,
        "target_state": "Objects near Earth's surface accelerate downward at approximately 9.8 m/s^2 when air resistance is neglected.",
        "target_question": "Is the acceleration approximately 20 m/s^2?",
        "irrelevant": "The object has mass 2 kg.",
        "distracting": "A different planet has gravitational acceleration approximately 20 m/s^2. The target is near Earth's surface."
    },
    {
        "case_id": "S05",
        "domain": "science_reasoning",
        "label": 1,
        "target_state": "Plants use carbon dioxide during photosynthesis.",
        "target_question": "Is carbon dioxide used during photosynthesis?",
        "irrelevant": "Plants also require water.",
        "distracting": "Plants release carbon dioxide during cellular respiration. The target process is photosynthesis."
    },
    {
        "case_id": "S06",
        "domain": "science_reasoning",
        "label": 0,
        "target_state": "Plants use carbon dioxide during photosynthesis.",
        "target_question": "Is carbon dioxide not used during photosynthesis?",
        "irrelevant": "Plants also require water.",
        "distracting": "Plants release carbon dioxide during cellular respiration. The target process is photosynthesis."
    },
    {
        "case_id": "S07",
        "domain": "science_reasoning",
        "label": 1,
        "target_state": "At constant pressure, heating an ideal gas increases its volume when the amount of gas is fixed.",
        "target_question": "Does heating increase the volume under these conditions?",
        "irrelevant": "The gas has a pressure of 1 atmosphere.",
        "distracting": "At constant volume, heating increases pressure instead. The target condition is constant pressure."
    },
    {
        "case_id": "S08",
        "domain": "science_reasoning",
        "label": 0,
        "target_state": "At constant pressure, heating an ideal gas increases its volume when the amount of gas is fixed.",
        "target_question": "Does heating decrease the volume under these conditions?",
        "irrelevant": "The gas is in a container.",
        "distracting": "At constant volume, heating increases pressure. The target condition is constant pressure."
    },
    {
        "case_id": "S09",
        "domain": "science_reasoning",
        "label": 1,
        "target_state": "An object with greater mass requires greater force to produce the same acceleration, assuming the same conditions.",
        "target_question": "Does greater mass require greater force for the same acceleration?",
        "irrelevant": "Both objects are moving horizontally.",
        "distracting": "A lighter object requires less force for the same acceleration. The target comparison concerns the heavier object."
    },
    {
        "case_id": "S10",
        "domain": "science_reasoning",
        "label": 0,
        "target_state": "An object with greater mass requires greater force to produce the same acceleration, assuming the same conditions.",
        "target_question": "Does greater mass require less force for the same acceleration?",
        "irrelevant": "Both objects are moving horizontally.",
        "distracting": "A lighter object requires less force for the same acceleration. The target comparison concerns the heavier object."
    },
	{
    	"case_id": "S11",
    	"domain": "science_reasoning",
    	"label": 1,
    	"target_state": "Light can propagate through a vacuum.",
    	"target_question": "Can light propagate through a vacuum?",
    	"irrelevant": "Sound requires a material medium to propagate.",
    	"distracting": "Sound cannot propagate through a vacuum, but light can. The target phenomenon is light.",
	},
    	{
    	"case_id": "S12",
    	"domain": "science_reasoning",
    	"label": 0,
    	"target_state": "Sound requires a material medium to propagate.",
    	"target_question": "Can sound propagate through a vacuum?",
    	"irrelevant": "Light can propagate through a vacuum.",
    	"distracting": "Light can propagate through a vacuum, but sound requires a material medium. The target phenomenon is sound.",
	},
    # -------------------------
    # DATA REASONING
    # -------------------------
    {
        "case_id": "D01",
        "domain": "data_reasoning",
        "label": 1,
        "target_state": "Dataset A contains the values 2, 4, 6, and 8.",
        "target_question": "Is the mean of Dataset A equal to 5?",
        "irrelevant": "Dataset B contains the values 100 and 200.",
        "distracting": "Dataset B has mean 150. The target is Dataset A, whose values are 2, 4, 6, and 8."
    },
    {
        "case_id": "D02",
        "domain": "data_reasoning",
        "label": 0,
        "target_state": "Dataset A contains the values 2, 4, 6, and 8.",
        "target_question": "Is the mean of Dataset A equal to 8?",
        "irrelevant": "Dataset B contains the values 8 and 8.",
        "distracting": "Dataset B has mean 8. The target is Dataset A, whose mean is 5."
    },
    {
        "case_id": "D03",
        "domain": "data_reasoning",
        "label": 1,
        "target_state": "Dataset A contains 20 observations, of which 5 belong to category X.",
        "target_question": "Is the proportion of category X equal to 25%?",
        "irrelevant": "Dataset B contains 100 observations.",
        "distracting": "Dataset B has category X proportion 80%. The target is Dataset A."
    },
    {
        "case_id": "D04",
        "domain": "data_reasoning",
        "label": 0,
        "target_state": "Dataset A contains 20 observations, of which 5 belong to category X.",
        "target_question": "Is the proportion of category X equal to 50%?",
        "irrelevant": "Dataset B contains 100 observations.",
        "distracting": "Dataset B has category X proportion 50%. The target is Dataset A."
    },
    {
        "case_id": "D05",
        "domain": "data_reasoning",
        "label": 1,
        "target_state": "Dataset A contains values 1, 3, 5, 7, and 9.",
        "target_question": "Is the median of Dataset A equal to 5?",
        "irrelevant": "Dataset B contains values 100 and 200.",
        "distracting": "Dataset B has median 150. The target is Dataset A."
    },
    {
        "case_id": "D06",
        "domain": "data_reasoning",
        "label": 0,
        "target_state": "Dataset A contains values 1, 3, 5, 7, and 9.",
        "target_question": "Is the median of Dataset A equal to 7?",
        "irrelevant": "Dataset B contains values 7 and 7.",
        "distracting": "Dataset B has median 7. The target is Dataset A, whose median is 5."
    },
    {
        "case_id": "D07",
        "domain": "data_reasoning",
        "label": 1,
        "target_state": "Dataset A has 80 correct predictions out of 100 predictions.",
        "target_question": "Is the accuracy of Dataset A's predictions 80%?",
        "irrelevant": "Dataset B has 90 correct predictions out of 100.",
        "distracting": "Dataset B has accuracy 90%. The target is Dataset A, whose accuracy is 80%."
    },
    {
        "case_id": "D08",
        "domain": "data_reasoning",
        "label": 0,
        "target_state": "Dataset A has 80 correct predictions out of 100 predictions.",
        "target_question": "Is the accuracy of Dataset A's predictions 90%?",
        "irrelevant": "Dataset B has 90 correct predictions out of 100.",
        "distracting": "Dataset B has accuracy 90%. The target is Dataset A, whose accuracy is 80%."
    },
    {
        "case_id": "D09",
        "domain": "data_reasoning",
        "label": 1,
        "target_state": "Group A contains 30 people, and 12 prefer option X.",
        "target_question": "Is the proportion preferring option X equal to 40%?",
        "irrelevant": "Group B contains 50 people.",
        "distracting": "Group B has 50% preference for option X. The target is Group A."
    },
    {
        "case_id": "D10",
        "domain": "data_reasoning",
        "label": 0,
        "target_state": "Group A contains 30 people, and 12 prefer option X.",
        "target_question": "Is the proportion preferring option X equal to 50%?",
        "irrelevant": "Group B contains 50 people.",
        "distracting": "Group B has 50% preference for option X. The target is Group A."
    },
    {
        "case_id": "D11",
        "domain": "data_reasoning",
        "label": 1,
        "target_state": "Dataset A contains values 2, 4, 8, 10, and 12.",
        "target_question": "Is the median of Dataset A equal to 8?",
        "irrelevant": "Dataset B contains values 8, 8, and 8.",
        "distracting": "Dataset B has median 8. The target is Dataset A, whose median is also 8."
    },
    {
        "case_id": "D12",
        "domain": "data_reasoning",
        "label": 0,
        "target_state": "Dataset A contains values 2, 4, 8, 10, and 12.",
        "target_question": "Is the median of Dataset A equal to 10?",
        "irrelevant": "Dataset B contains values 10, 10, and 10.",
        "distracting": "Dataset B has median 10. The target is Dataset A, whose median is 8."
    },
]

rows = []

for case in cases:
    base = {
        "case_id": case["case_id"],
        "domain": case["domain"],
        "label": case["label"],
    }

    rows.append({
        **base,
        "condition": "original",
        "state": case["target_state"],
        "question": case["target_question"],
    })

    rows.append({
        **base,
        "condition": "irrelevant_context",
        "state": (
            case["target_state"]
            + " "
            + case["irrelevant"]
        ),
        "question": case["target_question"],
    })

    rows.append({
        **base,
        "condition": "distracting_context",
        "state": (
            case["target_state"]
            + " "
            + case["distracting"]
        ),
        "question": case["target_question"],
    })

df = pd.DataFrame(rows)

assert len(cases) == 60
assert len(df) == 180
assert df["case_id"].nunique() == 60
assert df["label"].sum() == 90
assert (df["label"] == 0).sum() == 90
assert set(df["condition"]) == {
    "original",
    "irrelevant_context",
    "distracting_context",
}

for case_id, group in df.groupby("case_id"):
    assert len(group) == 3
    assert group["label"].nunique() == 1
    assert group["condition"].nunique() == 3

assert df.groupby("domain")["case_id"].nunique().eq(12).all()

out = Path("experiments/data/e9b_state_manipulation_dataset.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"Cases: {len(cases)}")
print(f"Rows: {len(df)}")
print(f"Labels: {df['label'].value_counts().to_dict()}")
print(f"Conditions: {df['condition'].value_counts().to_dict()}")
print(f"Domains: {df['domain'].value_counts().to_dict()}")
print(f"Saved: {out}")
