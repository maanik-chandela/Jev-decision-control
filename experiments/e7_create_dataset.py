import os
import pandas as pd


OUTPUT = "experiments/data/e7_language_shift_dataset.csv"


CASES = [
    # -------------------------
    # MATHEMATICS
    # -------------------------
    {
        "core_id": "M01",
        "domain": "mathematics",
        "label": 1,
        "english": "A square has side length 6 cm. Is its perimeter 24 cm?",
        "hindi": "एक वर्ग की भुजा की लंबाई 6 सेमी है। क्या उसका परिमाप 24 सेमी है?",
        "hinglish": "Ek square ki side ki length 6 cm hai. Kya uska perimeter 24 cm hai?"
    },
    {
        "core_id": "M02",
        "domain": "mathematics",
        "english": "A rectangle has length 8 cm and width 3 cm. Is its area 24 square cm?",
        "hindi": "एक आयत की लंबाई 8 सेमी और चौड़ाई 3 सेमी है। क्या उसका क्षेत्रफल 24 वर्ग सेमी है?",
        "hinglish": "Ek rectangle ki length 8 cm aur width 3 cm hai. Kya uska area 24 square cm hai?",
        "label": 1,
    },
    {
        "core_id": "M03",
        "domain": "mathematics",
        "label": 1,
        "english": "If x = 7, is 2x + 3 equal to 17?",
        "hindi": "यदि x = 7 है, तो क्या 2x + 3 का मान 17 है?",
        "hinglish": "Agar x = 7 hai, to kya 2x + 3 ka value 17 hai?"
    },
    {
        "core_id": "M04",
        "domain": "mathematics",
        "label": 0,
        "english": "The average of 4, 8, and 12 is 9. Is this statement true?",
        "hindi": "4, 8 और 12 का औसत 9 है। क्या यह कथन सही है?",
        "hinglish": "4, 8 aur 12 ka average 9 hai. Kya yeh statement sahi hai?"
    },
    {
        "core_id": "M05",
        "domain": "mathematics",
        "label": 1,
        "english": "Is 15 greater than 9?",
        "hindi": "क्या 15, 9 से बड़ा है?",
        "hinglish": "Kya 15, 9 se bada hai?"
    },
    {
        "core_id": "M06",
        "domain": "mathematics",
        "label": 0,
        "english": "Is 18 divisible by 5?",
        "hindi": "क्या 18, 5 से विभाज्य है?",
        "hinglish": "Kya 18, 5 se divisible hai?"
    },
    {
        "core_id": "M07",
        "domain": "mathematics",
        "label": 1,
        "english": "If a triangle has angles 60, 60, and 60 degrees, is it an equilateral triangle?",
        "hindi": "यदि किसी त्रिभुज के कोण 60, 60 और 60 डिग्री हैं, तो क्या वह समबाहु त्रिभुज है?",
        "hinglish": "Agar triangle ke angles 60, 60 aur 60 degree hain, to kya woh equilateral triangle hai?"
    },
    {
        "core_id": "M08",
        "domain": "mathematics",
        "label": 0,
        "english": "Is the square root of 49 equal to 6?",
        "hindi": "क्या 49 का वर्गमूल 6 के बराबर है?",
        "hinglish": "Kya 49 ka square root 6 ke barabar hai?"
    },
    {
        "core_id": "M09",
        "domain": "mathematics",
        "label": 1,
        "english": "Is 3 multiplied by 8 equal to 24?",
        "hindi": "क्या 3 को 8 से गुणा करने पर 24 प्राप्त होता है?",
        "hinglish": "Kya 3 ko 8 se multiply karne par 24 milta hai?"
    },
    {
        "core_id": "M10",
        "domain": "mathematics",
        "label": 0,
        "english": "Is 2 to the power of 5 equal to 30?",
        "hindi": "क्या 2 की घात 5 का मान 30 है?",
        "hinglish": "Kya 2 ki power 5 ka value 30 hai?"
    },
    {
        "core_id": "M11",
        "domain": "mathematics",
        "label": 1,
        "english": "Is 100 centimeters equal to 1 meter?",
        "hindi": "क्या 100 सेंटीमीटर, 1 मीटर के बराबर हैं?",
        "hinglish": "Kya 100 centimeters, 1 meter ke equal hain?"
    },
    {
        "core_id": "M12",
        "domain": "mathematics",
        "label": 0,
        "english": "A circle has radius 5 cm. Is its diameter 5 cm?",
        "hindi": "एक वृत्त की त्रिज्या 5 सेमी है। क्या उसका व्यास 5 सेमी है?",
        "hinglish": "Ek circle ka radius 5 cm hai. Kya uska diameter 5 cm hai?"
    },

    # -------------------------
    # PROBABILITY
    # -------------------------
    {
        "core_id": "P01",
        "domain": "probability",
        "label": 1,
        "english": "For a fair six-sided die, is the probability of rolling a 6 equal to 1/6?",
        "hindi": "एक निष्पक्ष छह-पक्षीय पासे के लिए, क्या 6 आने की प्रायिकता 1/6 है?",
        "hinglish": "Ek fair six-sided die ke liye, kya 6 aane ki probability 1/6 hai?"
    },
    {
        "core_id": "P02",
        "domain": "probability",
        "label": 1,
        "english": "For a fair six-sided die, is the probability of rolling an even number equal to 1/2?",
        "hindi": "एक निष्पक्ष छह-पक्षीय पासे के लिए, क्या सम संख्या आने की प्रायिकता 1/2 है?",
        "hinglish": "Ek fair six-sided die ke liye, kya even number aane ki probability 1/2 hai?"
    },
    {
        "core_id": "P03",
        "domain": "probability",
        "label": 1,
        "english": "For a fair coin, is the probability of heads equal to 0.5?",
        "hindi": "एक निष्पक्ष सिक्के के लिए, क्या हेड आने की प्रायिकता 0.5 है?",
        "hinglish": "Ek fair coin ke liye, kya heads aane ki probability 0.5 hai?"
    },
    {
        "core_id": "P04",
        "domain": "probability",
        "label": 0,
        "english": "For a fair coin, is the probability of tails equal to 0.8?",
        "hindi": "एक निष्पक्ष सिक्के के लिए, क्या टेल आने की प्रायिकता 0.8 है?",
        "hinglish": "Ek fair coin ke liye, kya tails aane ki probability 0.8 hai?"
    },
    {
        "core_id": "P05",
        "domain": "probability",
        "label": 1,
        "english": "If a bag contains 2 red balls and 2 blue balls, is the probability of drawing a red ball on the first draw 1/2?",
        "hindi": "यदि एक थैले में 2 लाल और 2 नीली गेंदें हैं, तो क्या पहली बार लाल गेंद निकालने की प्रायिकता 1/2 है?",
        "hinglish": "Agar bag mein 2 red aur 2 blue balls hain, to kya first draw mein red ball nikalne ki probability 1/2 hai?"
    },
    {
        "core_id": "P06",
        "domain": "probability",
        "label": 0,
        "english": "If a bag contains 3 red balls and 1 blue ball, is the probability of drawing a blue ball 1/2?",
        "hindi": "यदि एक थैले में 3 लाल और 1 नीली गेंद है, तो क्या नीली गेंद निकालने की प्रायिकता 1/2 है?",
        "hinglish": "Agar bag mein 3 red balls aur 1 blue ball hai, to kya blue ball nikalne ki probability 1/2 hai?"
    },
    {
        "core_id": "P07",
        "domain": "probability",
        "label": 1,
        "english": "For a fair six-sided die, is the probability of rolling a number greater than 4 equal to 1/3?",
        "hindi": "एक निष्पक्ष छह-पक्षीय पासे के लिए, क्या 4 से बड़ी संख्या आने की प्रायिकता 1/3 है?",
        "hinglish": "Ek fair six-sided die ke liye, kya 4 se badi number aane ki probability 1/3 hai?"
    },
    {
        "core_id": "P08",
        "domain": "probability",
        "label": 0,
        "english": "For a fair six-sided die, is the probability of rolling a number less than 3 equal to 1/2?",
        "hindi": "एक निष्पक्ष छह-पक्षीय पासे के लिए, क्या 3 से छोटी संख्या आने की प्रायिकता 1/2 है?",
        "hinglish": "Ek fair six-sided die ke liye, kya 3 se chhoti number aane ki probability 1/2 hai?"
    },
    {
        "core_id": "P09",
        "domain": "probability",
        "label": 1,
        "english": "In a finite sample space, if an event has probability 0, is the event 		impossible?",
	"hindi": "एक सीमित प्रतिदर्श समष्टि में, यदि किसी घटना की प्रायिकता 0 है, तो क्या वह घटना असंभव है?",
	"hinglish": "Ek finite sample space mein, agar kisi event ki probability 0 hai, to kya 		woh event impossible hai?"
    },
    {
        "core_id": "P10",
        "domain": "probability",
        "label": 0,
        "english": "If a probability is 1, is the event impossible?",
        "hindi": "यदि किसी घटना की प्रायिकता 1 है, तो क्या वह घटना असंभव है?",
        "hinglish": "Agar kisi event ki probability 1 hai, to kya woh event impossible hai?"
    },
    {
        "core_id": "P11",
        "domain": "probability",
        "label": 1,
        "english": "If two outcomes have probabilities 0.3 and 0.7 and are the only possible outcomes, do their probabilities sum to 1?",
        "hindi": "यदि दो परिणामों की प्रायिकताएँ 0.3 और 0.7 हैं और केवल यही दो परिणाम संभव हैं, तो क्या उनकी प्रायिकताओं का योग 1 है?",
        "hinglish": "Agar do outcomes ki probabilities 0.3 aur 0.7 hain aur sirf yehi do outcomes possible hain, to kya unki probabilities ka sum 1 hai?"
    },
    {
        "core_id": "P12",
        "domain": "probability",
        "label": 0,
        "english": "Can a probability of an event be greater than 1?",
        "hindi": "क्या किसी घटना की प्रायिकता 1 से अधिक हो सकती है?",
        "hinglish": "Kya kisi event ki probability 1 se greater ho sakti hai?"
    },

    # -------------------------
    # FORMAL LOGIC
    # -------------------------
    {
        "core_id": "L01",
        "domain": "formal_logic",
        "label": 1,
        "english": "If A implies B and A is true, must B be true?",
        "hindi": "यदि A से B निष्कर्षित होता है और A सत्य है, तो क्या B का सत्य होना आवश्यक है?",
        "hinglish": "Agar A se B imply hota hai aur A true hai, to kya B ka true hona zaroori hai?"
    },
    {
        "core_id": "L02",
        "domain": "formal_logic",
        "label": 0,
        "english": "If A implies B and B is true, must A be true?",
        "hindi": "यदि A से B निष्कर्षित होता है और B सत्य है, तो क्या A का सत्य होना आवश्यक है?",
        "hinglish": "Agar A se B imply hota hai aur B true hai, to kya A ka true hona zaroori hai?"
    },
    {
        "core_id": "L03",
        "domain": "formal_logic",
        "label": 0,
        "english": "If no cats are dogs and all poodles are dogs, can a poodle be a cat?",
        "hindi": "यदि कोई भी बिल्ली कुत्ता नहीं है और सभी पूडल कुत्ते हैं, तो क्या कोई पूडल बिल्ली हो सकता है?",
        "hinglish": "Agar koi cat dog nahi hai aur saare poodles dogs hain, to kya koi poodle cat ho sakta hai?"
    },
    {
        "core_id": "L04",
        "domain": "formal_logic",
        "label": 0,
        "english": "If all A are B and some B are C, must some A be C?",
        "hindi": "यदि सभी A, B हैं और कुछ B, C हैं, तो क्या यह आवश्यक है कि कुछ A, C हों?",
        "hinglish": "Agar saare A, B hain aur kuch B, C hain, to kya zaroori hai ki kuch A, C hon?"
    },
    {
        "core_id": "L05",
        "domain": "formal_logic",
        "label": 0,
        "english": "If A is false and A implies B, does this guarantee that B is false?",
        "hindi": "यदि A असत्य है और A से B निष्कर्षित होता है, तो क्या इससे यह सुनिश्चित होता है कि B असत्य है?",
        "hinglish": "Agar A false hai aur A se B imply hota hai, to kya isse B ka false hona guarantee hota hai?"
    },
    {
        "core_id": "L06",
        "domain": "formal_logic",
        "label": 0,
        "english": "If A implies B and B is false, can A be true?",
        "hindi": "यदि A से B निष्कर्षित होता है और B असत्य है, तो क्या A सत्य हो सकता है?",
        "hinglish": "Agar A se B imply hota hai aur B false hai, to kya A true ho sakta hai?"
    },
    {
        "core_id": "L07",
        "domain": "formal_logic",
        "label": 1,
        "english": "If every square is a rectangle, is every square a rectangle?",
        "hindi": "यदि प्रत्येक वर्ग एक आयत है, तो क्या प्रत्येक वर्ग एक आयत है?",
        "hinglish": "Agar har square ek rectangle hai, to kya har square ek rectangle hai?"
    },
    {
        "core_id": "L08",
        "domain": "formal_logic",
        "label": 0,
        "english": "If some A are B, does that mean every A is B?",
        "hindi": "यदि कुछ A, B हैं, तो क्या इसका अर्थ है कि प्रत्येक A, B है?",
        "hinglish": "Agar kuch A, B hain, to kya iska matlab hai ki har A, B hai?"
    },
    {
        "core_id": "L09",
        "domain": "formal_logic",
        "label": 0,
        "english": "If no mammals are insects and all whales are mammals, can a whale be an insect?",
        "hindi": "यदि कोई भी स्तनधारी कीट नहीं है और सभी व्हेल स्तनधारी हैं, तो क्या कोई व्हेल कीट हो सकती है?",
        "hinglish": "Agar koi mammal insect nahi hai aur saari whales mammals hain, to kya koi whale insect ho sakti hai?"
    },
    {
        "core_id": "L10",
        "domain": "formal_logic",
        "label": 0,
        "english": "If all A are B and all B are C, must every C be A?",
        "hindi": "यदि सभी A, B हैं और सभी B, C हैं, तो क्या प्रत्येक C का A होना आवश्यक है?",
        "hinglish": "Agar saare A, B hain aur saare B, C hain, to kya har C ka A hona zaroori hai?"
    },
    {
        "core_id": "L11",
        "domain": "formal_logic",
        "label": 1,
        "english": "If an object is either red or blue, and it is not red, must it be blue?",
        "hindi": "यदि कोई वस्तु या तो लाल या नीली है और वह लाल नहीं है, तो क्या उसका नीला होना आवश्यक है?",
        "hinglish": "Agar object ya to red ya blue hai aur woh red nahi hai, to kya uska blue hona zaroori hai?"
    },
    {
        "core_id": "L12",
        "domain": "formal_logic",
        "label": 0,
        "english": "If an object is either red or blue, and it is red, must it be green?",
        "hindi": "यदि कोई वस्तु या तो लाल या नीली है और वह लाल है, तो क्या उसका हरा होना आवश्यक है?",
        "hinglish": "Agar object ya to red ya blue hai aur woh red hai, to kya uska green hona zaroori hai?"
    },

    # -------------------------
    # SCIENCE REASONING
    # -------------------------
    {
        "core_id": "S01",
        "domain": "science_reasoning",
        "label": 1,
        "english": "At standard atmospheric pressure, does water boil at about 100 degrees Celsius?",
        "hindi": "मानक वायुमंडलीय दाब पर, क्या पानी लगभग 100 डिग्री सेल्सियस पर उबलता है?",
        "hinglish": "Standard atmospheric pressure par, kya water lagbhag 100 degree Celsius par boil hota hai?"
    },
    {
        "core_id": "S02",
        "domain": "science_reasoning",
        "label": 0,
        "english": "At standard atmospheric pressure, does water boil at about 50 degrees Celsius?",
        "hindi": "मानक वायुमंडलीय दाब पर, क्या पानी लगभग 50 डिग्री सेल्सियस पर उबलता है?",
        "hinglish": "Standard atmospheric pressure par, kya water lagbhag 50 degree Celsius par boil hota hai?"
    },
    {
        "core_id": "S03",
        "domain": "science_reasoning",
        "label": 1,
        "english": "Does Earth orbit the Sun?",
        "hindi": "क्या पृथ्वी सूर्य की परिक्रमा करती है?",
        "hinglish": "Kya Earth Sun ke around orbit karti hai?"
    },
    {
        "core_id": "S04",
        "domain": "science_reasoning",
        "label": 0,
        "english": "Does the Moon produce most of its own visible light?",
        "hindi": "क्या चंद्रमा अपने दिखाई देने वाले अधिकांश प्रकाश का स्वयं उत्पादन करता है?",
        "hinglish": "Kya Moon apni visible light ka most hissa khud produce karta hai?"
    },
    {
        "core_id": "S05",
        "domain": "science_reasoning",
        "label": 1,
        "english": "Does increasing temperature generally increase the average kinetic energy of particles in a substance?",
        "hindi": "क्या किसी पदार्थ का तापमान बढ़ाने से सामान्यतः उसके कणों की औसत गतिज ऊर्जा बढ़ती है?",
        "hinglish": "Kya kisi substance ka temperature badhane se generally particles ki average kinetic energy badhti hai?"
    },
    {
        "core_id": "S06",
        "domain": "science_reasoning",
        "label": 0,
        "english": "Does increasing temperature generally decrease the average kinetic energy of particles in a substance?",
        "hindi": "क्या किसी पदार्थ का तापमान बढ़ाने से सामान्यतः उसके कणों की औसत गतिज ऊर्जा घटती है?",
        "hinglish": "Kya kisi substance ka temperature badhane se generally particles ki average kinetic energy kam hoti hai?"
    },
    {
        "core_id": "S07",
        "domain": "science_reasoning",
        "label": 1,
        "english": "Does photosynthesis allow green plants to convert light energy into stored chemical energy?",
        "hindi": "क्या प्रकाश संश्लेषण हरे पौधों को प्रकाश ऊर्जा को संचित रासायनिक ऊर्जा में बदलने देता है?",
        "hinglish": "Kya photosynthesis green plants ko light energy ko stored chemical energy mein convert karne deta hai?"
    },
    {
        "core_id": "S08",
        "domain": "science_reasoning",
        "label": 0,
        "english": "Does photosynthesis allow green plants to convert chemical energy directly into sunlight?",
        "hindi": "क्या प्रकाश संश्लेषण हरे पौधों को रासायनिक ऊर्जा को सीधे सूर्य के प्रकाश में बदलने देता है?",
        "hinglish": "Kya photosynthesis green plants ko chemical energy ko directly sunlight mein convert karne deta hai?"
    },
    {
        "core_id": "S09",
        "domain": "science_reasoning",
        "label": 1,
        "english": "Is carbon dioxide a gas at ordinary room conditions?",
        "hindi": "क्या सामान्य कमरे की परिस्थितियों में कार्बन डाइऑक्साइड एक गैस है?",
        "hinglish": "Kya normal room conditions mein carbon dioxide ek gas hai?"
    },
    {
        "core_id": "S10",
        "domain": "science_reasoning",
        "label": 0,
        "english": "Is carbon dioxide a liquid at ordinary room conditions?",
        "hindi": "क्या सामान्य कमरे की परिस्थितियों में कार्बन डाइऑक्साइड एक तरल है?",
        "hinglish": "Kya normal room conditions mein carbon dioxide ek liquid hai?"
    },
    {
        "core_id": "S11",
        "domain": "science_reasoning",
        "label": 1,
        "english": "Does a vacuum contain no matter in the idealized physical sense?",
        "hindi": "क्या आदर्श भौतिक अर्थ में निर्वात में कोई पदार्थ नहीं होता?",
        "hinglish": "Idealized physical sense mein, kya vacuum mein koi matter nahi hota?"
    },
    {
        "core_id": "S12",
        "domain": "science_reasoning",
        "label": 0,
        "english": "Does sound normally travel faster through air than through steel?",
        "hindi": "क्या ध्वनि सामान्यतः स्टील की तुलना में हवा में अधिक तेजी से चलती है?",
        "hinglish": "Kya sound normally steel ke comparison mein air mein faster travel karti hai?"
    },

    # -------------------------
    # DATA REASONING
    # -------------------------
    {
        "core_id": "D01",
        "domain": "data_reasoning",
        "label": 1,
        "english": "A dataset contains 10 values and 4 of them are labeled positive. Is the positive proportion 40%?",
        "hindi": "एक डेटासेट में 10 मान हैं और उनमें से 4 को सकारात्मक लेबल दिया गया है। क्या सकारात्मक अनुपात 40% है?",
        "hinglish": "Ek dataset mein 10 values hain aur unmein se 4 positive labeled hain. Kya positive proportion 40% hai?"
    },
    {
        "core_id": "D02",
        "domain": "data_reasoning",
        "label": 1,
        "english": "A dataset contains 10 values and 3 of them are labeled positive. Is the positive proportion 30%?",
        "hindi": "एक डेटासेट में 10 मान हैं और उनमें से 3 को सकारात्मक लेबल दिया गया है। क्या सकारात्मक अनुपात 30% है?",
        "hinglish": "Ek dataset mein 10 values hain aur unmein se 3 positive labeled hain. Kya positive proportion 30% hai?"
    },
    {
        "core_id": "D03",
        "domain": "data_reasoning",
        "label": 1,
        "english": "A classifier correctly predicts 80 out of 100 examples. Is its accuracy 80%?",
        "hindi": "एक क्लासिफायर 100 उदाहरणों में से 80 का सही अनुमान लगाता है। क्या उसकी सटीकता 80% है?",
        "hinglish": "Ek classifier 100 examples mein se 80 ko correctly predict karta hai. Kya uski accuracy 80% hai?"
    },
    {
        "core_id": "D04",
        "domain": "data_reasoning",
        "label": 0,
        "english": "A classifier correctly predicts 70 out of 100 examples. Is its accuracy 80%?",
        "hindi": "एक क्लासिफायर 100 उदाहरणों में से 70 का सही अनुमान लगाता है। क्या उसकी सटीकता 80% है?",
        "hinglish": "Ek classifier 100 examples mein se 70 ko correctly predict karta hai. Kya uski accuracy 80% hai?"
    },
    {
        "core_id": "D05",
        "domain": "data_reasoning",
        "label": 1,
        "english": "A list contains 2, 4, 6, 8, and 10. Is its median 6?",
        "hindi": "एक सूची में 2, 4, 6, 8 और 10 हैं। क्या उसका माध्यिका 6 है?",
        "hinglish": "Ek list mein 2, 4, 6, 8 aur 10 hain. Kya uska median 6 hai?"
    },
    {
        "core_id": "D06",
        "domain": "data_reasoning",
        "label": 0,
        "english": "A list contains 2, 4, 6, 8, and 10. Is its median 8?",
        "hindi": "एक सूची में 2, 4, 6, 8 और 10 हैं। क्या उसका माध्यिका 8 है?",
        "hinglish": "Ek list mein 2, 4, 6, 8 aur 10 hain. Kya uska median 8 hai?"
    },
    {
        "core_id": "D07",
        "domain": "data_reasoning",
        "label": 1,
        "english": "A sample has 25 observations and 5 are missing. Are 20% of the observations missing?",
        "hindi": "एक नमूने में 25 अवलोकन हैं और 5 गायब हैं। क्या 20% अवलोकन गायब हैं?",
        "hinglish": "Ek sample mein 25 observations hain aur 5 missing hain. Kya 20% observations missing hain?"
    },
    {
        "core_id": "D08",
        "domain": "data_reasoning",
        "label": 0,
        "english": "A sample has 25 observations and 10 are missing. Are 20% of the observations missing?",
        "hindi": "एक नमूने में 25 अवलोकन हैं और 10 गायब हैं। क्या 20% अवलोकन गायब हैं?",
        "hinglish": "Ek sample mein 25 observations hain aur 10 missing hain. Kya 20% observations missing hain?"
    },
    {
        "core_id": "D09",
        "domain": "data_reasoning",
        "label": 1,
        "english": "If a model has 90 correct predictions out of 100, is its error rate 10%?",
        "hindi": "यदि किसी मॉडल के 100 में से 90 पूर्वानुमान सही हैं, तो क्या उसकी त्रुटि दर 10% है?",
        "hinglish": "Agar model ke 100 mein se 90 predictions correct hain, to kya uski error rate 10% hai?"
    },
    {
        "core_id": "D10",
        "domain": "data_reasoning",
        "label": 0,
        "english": "If a model has 90 correct predictions out of 100, is its error rate 20%?",
        "hindi": "यदि किसी मॉडल के 100 में से 90 पूर्वानुमान सही हैं, तो क्या उसकी त्रुटि दर 20% है?",
        "hinglish": "Agar model ke 100 mein se 90 predictions correct hain, to kya uski error rate 20% hai?"
    },
    {
        "core_id": "D11",
        "domain": "data_reasoning",
        "label": 1,
        "english": "If a dataset has 50 examples and 25 belong to class A, is class A exactly half of the dataset?",
        "hindi": "यदि किसी डेटासेट में 50 उदाहरण हैं और 25 क्लास A से संबंधित हैं, तो क्या क्लास A डेटासेट का ठीक आधा है?",
        "hinglish": "Agar dataset mein 50 examples hain aur 25 class A ke hain, to kya class A dataset ka exactly half hai?"
    },
    {
        "core_id": "D12",
        "domain": "data_reasoning",
        "label": 0,
        "english": "If a dataset has 50 examples and 15 belong to class A, is class A exactly half of the dataset?",
        "hindi": "यदि किसी डेटासेट में 50 उदाहरण हैं और 15 क्लास A से संबंधित हैं, तो क्या क्लास A डेटासेट का ठीक आधा है?",
        "hinglish": "Agar dataset mein 50 examples hain aur 15 class A ke hain, to kya class A dataset ka exactly half hai?"
    },
]


def validate_cases():
    assert len(CASES) == 60
    assert sum(c["label"] for c in CASES) == 30

    ids = [c["core_id"] for c in CASES]
    assert len(ids) == len(set(ids))

    domains = pd.Series([c["domain"] for c in CASES]).value_counts()

    expected_domains = {
        "mathematics": 12,
        "probability": 12,
        "formal_logic": 12,
        "science_reasoning": 12,
        "data_reasoning": 12,
    }

    assert domains.to_dict() == expected_domains

    for c in CASES:
        assert c["english"]
        assert c["hindi"]
        assert c["hinglish"]
        assert c["label"] in [0, 1]

        # All three versions should be statements/questions,
        # not explanations or answers.
        for key in ["english", "hindi", "hinglish"]:
            assert len(c[key]) > 10


def build_dataset():
    rows = []

    for case in CASES:
        for language, question in [
            ("english", case["english"]),
            ("hindi", case["hindi"]),
            ("hinglish", case["hinglish"]),
        ]:
            suffix = {
                "english": "EN",
                "hindi": "HI",
                "hinglish": "HG",
            }[language]

            rows.append(
                {
                    "case_id": f"E7_{case['core_id']}_{suffix}",
                    "core_id": case["core_id"],
                    "domain": case["domain"],
                    "language": language,
                    "label": case["label"],
                    "question": question,
                }
            )

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved: {OUTPUT}")
    print(f"Rows: {len(df)}")
    print("\nLanguage counts:")
    print(df["language"].value_counts())

    print("\nLabel counts:")
    print(df.groupby("language")["label"].value_counts().unstack(fill_value=0))

    print("\nDomain counts:")
    print(df.groupby(["language", "domain"]).size())


if __name__ == "__main__":
    validate_cases()
    build_dataset()

