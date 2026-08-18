#!/usr/bin/env python3
# audit prompt bank
#
# ~330 prompts across knowledge, math, reasoning, code, instruction following,
# writing and conversation. each carries a check spec so most can be scored
# automatically; the rest get read by analysis agents. `grid=True` marks the
# ~60-prompt subset used for the sampling-parameter grid.

# check kinds:
#   contains   -> ok if ANY of expect (case-insensitive) appears
#   number     -> ok if expect appears as a standalone number
#   count      -> ok if response enumerates exactly expect items (list Qs)
#   exec       -> python code, scored offline via ast+exec harness
#   word_limit -> ok if response has <= expect words
#   yesno      -> ok if response starts with expected yes/no
#   none       -> read by agents, no auto score

P = []

def add(cat, q, check="none", expect=None, diff="easy", grid=False):
    P.append({
        "id": f"{cat}_{len([p for p in P if p['category'] == cat]):03d}",
        "category": cat, "prompt": q, "check": check,
        "expect": expect, "difficulty": diff, "grid": grid,
    })

# ── factual: capitals ────────────────────────────────────────
CAPITALS = [
    ("France", ["Paris"]), ("Japan", ["Tokyo"]), ("Australia", ["Canberra"]),
    ("Canada", ["Ottawa"]), ("Germany", ["Berlin"]), ("Italy", ["Rome"]),
    ("Spain", ["Madrid"]), ("Russia", ["Moscow"]), ("China", ["Beijing"]),
    ("India", ["New Delhi", "Delhi"]), ("Brazil", ["Brasilia", "Brasília"]),
    ("Egypt", ["Cairo"]), ("Mexico", ["Mexico City"]), ("Turkey", ["Ankara"]),
    ("Poland", ["Warsaw"]), ("Greece", ["Athens"]), ("Portugal", ["Lisbon"]),
    ("Argentina", ["Buenos Aires"]), ("South Korea", ["Seoul"]),
    ("the United Kingdom", ["London"]),
]
for i, (c, a) in enumerate(CAPITALS):
    add("capitals", f"What is the capital of {c}?", "contains", a,
        grid=(i < 4))

# ── factual: people ──────────────────────────────────────────
PEOPLE = [
    ("Donald Trump", ["president"]), ("Barack Obama", ["president"]),
    ("George Washington", ["president", "first"]),
    ("Abraham Lincoln", ["president"]),
    ("Albert Einstein", ["physicist", "relativity", "scientist"]),
    ("Isaac Newton", ["physicist", "gravity", "scientist", "mathematician"]),
    ("William Shakespeare", ["playwright", "poet", "writer"]),
    ("Beyonce", ["singer", "musician", "artist"]),
    ("Taylor Swift", ["singer", "musician", "artist"]),
    ("Lionel Messi", ["soccer", "football"]),
    ("Cristiano Ronaldo", ["soccer", "football"]),
    ("Martin Luther King", ["civil rights", "activist", "minister"]),
    ("Napoleon", ["french", "emperor", "general"]),
    ("Nelson Mandela", ["south africa", "president", "apartheid"]),
    ("Leonardo da Vinci", ["artist", "painter", "inventor", "mona lisa"]),
    ("Mozart", ["composer", "music"]),
    ("Pablo Picasso", ["painter", "artist"]),
    ("Elon Musk", ["tesla", "spacex", "entrepreneur", "businessman"]),
    ("Bill Gates", ["microsoft"]),
    ("Steve Jobs", ["apple"]),
    ("Michael Jordan", ["basketball"]),
    ("Serena Williams", ["tennis"]),
    ("Neil Armstrong", ["astronaut", "moon"]),
    ("Marie Curie", ["scientist", "physicist", "chemist", "radium", "radioactivity"]),
    ("Cleopatra", ["egypt", "queen", "pharaoh"]),
]
for i, (who, kw) in enumerate(PEOPLE):
    add("people", f"Who is {who}?" if who in ("Donald Trump", "Beyonce",
        "Taylor Swift", "Lionel Messi", "Cristiano Ronaldo", "Elon Musk",
        "Bill Gates", "Michael Jordan", "Serena Williams")
        else f"Who was {who}?", "contains", kw, grid=(i < 4))

# ── factual: science ─────────────────────────────────────────
SCI = [
    ("What is the chemical formula for water?", ["H2O"]),
    ("At what temperature does water boil at sea level in Celsius?", ["100"]),
    ("How many planets are in our solar system?", ["eight", "8"]),
    ("What is the largest planet in our solar system?", ["Jupiter"]),
    ("What is the closest star to Earth?", ["sun"]),
    ("How many chambers does the human heart have?", ["four", "4"]),
    ("What does DNA stand for?", ["deoxyribonucleic"]),
    ("What gas do plants absorb during photosynthesis?", ["carbon dioxide", "CO2"]),
    ("What are the three states of matter?", ["solid"]),
    ("What is the smallest unit of life?", ["cell"]),
    ("What is the chemical symbol for oxygen?", ["O"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("Is the sun a star or a planet?", ["star"]),
    ("What does the moon orbit?", ["earth"]),
    ("Which travels faster, light or sound?", ["light"]),
    ("At what temperature does water freeze in Celsius?", ["0", "zero"]),
    ("How many legs does an insect have?", ["six", "6"]),
    ("How many legs does a spider have?", ["eight", "8"]),
    ("What is the largest ocean on Earth?", ["Pacific"]),
    ("What is the longest river in the world?", ["Nile", "Amazon"]),
    ("What is the tallest mountain on Earth?", ["Everest"]),
    ("What organ pumps blood around the body?", ["heart"]),
    ("What do bees collect from flowers?", ["nectar", "pollen"]),
    ("What force pulls objects toward the Earth?", ["gravity"]),
    ("What planet is known as the Red Planet?", ["Mars"]),
]
for i, (q, kw) in enumerate(SCI):
    add("science", q, "contains", kw, grid=(i < 4))

# ── factual: geography & history ─────────────────────────────
GEO = [
    ("In what year did World War II end?", ["1945"]),
    ("Who was the first president of the United States?", ["Washington"]),
    ("In what country are the pyramids of Giza?", ["Egypt"]),
    ("In what country is the Great Wall?", ["China"]),
    ("In what city is the Eiffel Tower?", ["Paris"]),
    ("In what city is the Statue of Liberty?", ["New York"]),
    ("In what year did the Titanic sink?", ["1912"]),
    ("In what year did humans first land on the moon?", ["1969"]),
    ("In what year did the United States declare independence?", ["1776"]),
    ("What is the largest country in the world by area?", ["Russia"]),
    ("What is the most populous country in the world?", ["India", "China"]),
    ("What language is spoken in Brazil?", ["Portuguese"]),
    ("What language is spoken in Mexico?", ["Spanish"]),
    ("How many states are in the United States?", ["50", "fifty"]),
    ("On what continent is Egypt?", ["Africa"]),
    ("On what continent is Brazil?", ["South America"]),
    ("What ocean is between America and Europe?", ["Atlantic"]),
    ("What country has the maple leaf on its flag?", ["Canada"]),
    ("What is the currency of the United States?", ["dollar"]),
    ("What is the currency of Japan?", ["yen"]),
]
for i, (q, kw) in enumerate(GEO):
    add("geo_hist", q, "contains", kw, grid=(i < 3))

# ── lists ────────────────────────────────────────────────────
LISTS = [
    ("What are the 7 continents?", 7,
     ["asia", "africa", "europe", "australia", "antarctica"]),
    ("What are the three primary colors?", 3, ["red", "blue", "yellow"]),
    ("What are the four seasons?", 4, ["spring", "summer", "fall", "autumn", "winter"]),
    ("What are the days of the week?", 7, ["monday", "sunday"]),
    ("Name the planets in our solar system.", 8, ["mercury", "jupiter", "neptune"]),
    ("What are the five oceans?", 5, ["pacific", "atlantic", "indian", "arctic"]),
    ("What are the colors of the rainbow?", 7, ["red", "violet"]),
    ("What are the five senses?", 5, ["sight", "hearing", "touch", "taste", "smell"]),
    ("Name the four cardinal directions.", 4, ["north", "south", "east", "west"]),
    ("Name three fruits.", 3, []),
    ("Name three mammals.", 3, []),
    ("Name three countries in Europe.", 3, []),
    ("List the vowels in the English alphabet.", 5, ["a", "e", "i", "o", "u"]),
    ("Name three musical instruments.", 3, []),
    ("What are the two main political parties in the United States?", 2,
     ["democrat", "republican"]),
]
for i, (q, n, kw) in enumerate(LISTS):
    add("lists", q, "count", {"n": n, "kw": kw}, grid=(i < 5))

# ── arithmetic ───────────────────────────────────────────────
ADDS = [(85, 34), (53, 88), (17, 49), (26, 58), (91, 12), (44, 37),
        (68, 25), (73, 19), (36, 47), (59, 22)]
for i, (a, b) in enumerate(ADDS):
    add("math", f"What is {a} + {b}?", "number", str(a + b), grid=(i < 3))
SUBS = [(90, 47), (63, 28), (81, 39), (55, 17), (72, 45)]
for i, (a, b) in enumerate(SUBS):
    add("math", f"What is {a} - {b}?", "number", str(a - b), grid=(i < 2))
MULS = [(7, 8), (6, 9), (12, 4), (9, 9), (8, 6)]
for i, (a, b) in enumerate(MULS):
    add("math", f"What is {a} times {b}?", "number", str(a * b), grid=(i < 2))
DIVS = [(48, 6), (81, 9), (56, 8), (100, 4), (63, 7)]
for i, (a, b) in enumerate(DIVS):
    add("math", f"What is {a} divided by {b}?", "number", str(a // b), grid=(i < 1))
for i, (q, ans, d) in enumerate([
    ("If a car travels at 60 miles per hour for 3 hours, how far does it travel?", "180", "medium"),
    ("If a train travels at 72 miles per hour for 2 hours, how far does it go?", "144", "medium"),
    ("Apples cost 2 dollars each. How much do 5 apples cost?", "10", "easy"),
    ("A book costs 7 dollars. How much do 3 books cost?", "21", "easy"),
    ("I have 24 cookies and share them equally among 6 friends. How many does each friend get?", "4", "medium"),
    ("There are 12 eggs in a dozen. How many eggs are in 3 dozen?", "36", "medium"),
    ("Sarah has 15 marbles and gives away 6. How many does she have left?", "9", "easy"),
    ("What is half of 90?", "45", "easy"),
    ("What is double 35?", "70", "easy"),
    ("What is 10 percent of 200?", "20", "medium"),
]):
    add("math_word", q, "number", ans, diff=d, grid=(i < 3))

# ── reasoning ────────────────────────────────────────────────
REASON = [
    ("If all cats are animals, and Tom is a cat, is Tom an animal? Answer yes or no.", "yesno", "yes", "easy"),
    ("Which is heavier, a kilogram of feathers or a kilogram of rocks?", "contains", ["same", "equal", "neither"], "medium"),
    ("If today is Monday, what day is tomorrow?", "contains", ["Tuesday"], "easy"),
    ("If today is Friday, what day was yesterday?", "contains", ["Thursday"], "easy"),
    ("What is the opposite of hot?", "contains", ["cold"], "easy"),
    ("What is the opposite of up?", "contains", ["down"], "easy"),
    ("What comes next in the sequence: 2, 4, 6, 8?", "contains", ["10", "ten"], "easy"),
    ("What comes next in the sequence: 5, 10, 15, 20?", "contains", ["25"], "easy"),
    ("Anna is taller than Ben. Ben is taller than Carl. Who is tallest? ", "contains", ["Anna"], "medium"),
    ("A rooster lays an egg on a roof. Which way does the egg roll?", "contains", ["rooster", "don't lay", "do not lay", "no egg"], "hard"),
    ("Which is bigger, an elephant or a mouse?", "contains", ["elephant"], "easy"),
    ("Is a whale a fish or a mammal?", "contains", ["mammal"], "medium"),
    ("Can a person be their own grandfather? Answer yes or no with one reason.", "none", None, "hard"),
    ("If you drop a glass on a stone floor, what will likely happen?", "contains", ["break", "shatter", "crack"], "easy"),
    ("Tom is older than Jane. Who is younger?", "contains", ["Jane"], "easy"),
    ("Which weighs more, 2 kilograms or 1500 grams?", "contains", ["2 kilograms", "2 kg", "two kilograms"], "medium"),
    ("If a doctor gives you three pills and tells you to take one every half hour, how long until they are all taken?", "contains", ["hour", "60"], "hard"),
    ("A farmer has 17 sheep and all but 9 die. How many are left?", "contains", ["9", "nine"], "hard"),
    ("What has hands but cannot clap?", "contains", ["clock"], "medium"),
    ("If you are running a race and pass the person in second place, what place are you in?", "contains", ["second"], "hard"),
]
for i, (q, ck, kw, d) in enumerate(REASON):
    add("reasoning", q, ck, kw, diff=d, grid=(i < 5))

# ── code ─────────────────────────────────────────────────────
CODE = [
    ("Write a Python function called add_numbers that takes two numbers as input and returns their sum.",
     "exec", {"fn": "add_numbers", "tests": [((2, 3), 5), ((-1, 1), 0), ((10, 25), 35)]}, "easy"),
    ("Write a Python function called is_even(n) that returns True if n is even and False otherwise.",
     "exec", {"fn": "is_even", "tests": [((4,), True), ((7,), False), ((0,), True)]}, "easy"),
    ("Write a Python function called reverse_string(s) that returns the string reversed.",
     "exec", {"fn": "reverse_string", "tests": [(("abc",), "cba"), (("hello",), "olleh")]}, "easy"),
    ("Write a Python function called square(n) that returns n multiplied by itself.",
     "exec", {"fn": "square", "tests": [((3,), 9), ((-2,), 4)]}, "easy"),
    ("Write a Python function called max_of_two(a, b) that returns the larger of the two numbers.",
     "exec", {"fn": "max_of_two", "tests": [((3, 7), 7), ((10, 2), 10)]}, "easy"),
    ("Write a Python function called greet(name) that returns the string 'Hello, ' followed by the name.",
     "exec", {"fn": "greet", "tests": [(("Sam",), "Hello, Sam")]}, "medium"),
    ("Write a Python function called sum_list(numbers) that returns the sum of a list of numbers.",
     "exec", {"fn": "sum_list", "tests": [(([1, 2, 3],), 6), (([],), 0)]}, "medium"),
    ("Write a Python function called count_vowels(s) that returns how many vowels are in the string s.",
     "exec", {"fn": "count_vowels", "tests": [(("hello",), 2), (("xyz",), 0)]}, "medium"),
    ("Write a Python function called factorial(n) that returns n factorial.",
     "exec", {"fn": "factorial", "tests": [((0,), 1), ((5,), 120)]}, "hard"),
    ("Write a Python function called celsius_to_fahrenheit(c) that converts Celsius to Fahrenheit.",
     "exec", {"fn": "celsius_to_fahrenheit", "tests": [((0,), 32.0), ((100,), 212.0)]}, "medium"),
]
for i, (q, ck, spec, d) in enumerate(CODE):
    add("code", q, ck, spec, diff=d, grid=(i < 4))
for q, kw in [
    ("What does the print function do in Python?", ["output", "display", "print", "console", "screen"]),
    ("In Python, what is the result of 2 + 2?", ["4", "four"]),
    ("What symbol starts a comment in Python?", ["#", "hash", "pound"]),
    ("In Python, what keyword defines a function?", ["def"]),
    ("What does len(\"hello\") return in Python?", ["5", "five"]),
]:
    add("code_qa", q, "contains", kw, diff="easy")

# ── instruction following ────────────────────────────────────
INSTR = [
    ("What is the capital of France? Answer with one word only.", "word_limit", 3, "easy"),
    ("What color is the sky on a clear day? Answer with one word only.", "word_limit", 3, "easy"),
    ("Is fire hot? Answer only yes or no.", "word_limit", 3, "easy"),
    ("Name exactly three animals. Just list them, nothing else.", "count", {"n": 3, "kw": []}, "easy"),
    ("Name exactly two colors. Just the colors, no explanation.", "count", {"n": 2, "kw": []}, "easy"),
    ("Count from 1 to 5.", "contains", ["1, 2, 3, 4, 5", "1 2 3 4 5", "one, two, three, four, five"], "easy"),
    ("Say the word 'hello' and nothing else.", "word_limit", 2, "easy"),
    ("Answer in one short sentence: why do we sleep?", "word_limit", 25, "medium"),
    ("In exactly one sentence, describe the ocean.", "word_limit", 30, "medium"),
    ("Is the Earth flat? Answer only yes or no.", "yesno", "no", "easy"),
    ("Is water wet? Answer only yes or no.", "yesno", "yes", "easy"),
    ("Do penguins fly? Answer only yes or no.", "yesno", "no", "medium"),
    ("Give me exactly three tips for studying. Number them 1, 2, 3.", "count", {"n": 3, "kw": []}, "medium"),
    ("What is 2 plus 2? Reply with just the number.", "word_limit", 2, "easy"),
    ("Translate 'hello' into Spanish. One word answer.", "contains", ["hola"], "medium"),
]
for i, (q, ck, spec, d) in enumerate(INSTR):
    add("instruction", q, ck, spec, diff=d, grid=(i < 6))

# ── writing ──────────────────────────────────────────────────
WRITE = [
    "Write two sentences about the moon.",
    "Write a short paragraph about why exercise is good for you.",
    "Write a three-sentence story about a lost dog finding its way home.",
    "Describe a rainy day in two sentences.",
    "Write a short thank-you note to a teacher.",
    "Write one sentence that uses the word 'lighthouse'.",
    "Write a short paragraph explaining why reading matters.",
    "Describe your favorite season in a few sentences.",
    "Write the first sentence of a mystery story.",
    "Write two sentences describing a busy city street.",
]
for i, q in enumerate(WRITE):
    add("writing", q, "none", None, grid=(i < 3))

# ── conversation & identity ──────────────────────────────────
CONVO = [
    ("Hello! How are you today?", "none", None),
    ("Who are you?", "none", None),
    ("What is your name?", "none", None),
    ("Who made you?", "none", None),
    ("What can you help me with?", "none", None),
    ("Are you ChatGPT?", "none", None),
    ("Are you a human?", "contains", ["no", "not", "AI", "model", "assistant"]),
    ("Thanks for your help!", "none", None),
    ("Good morning!", "none", None),
    ("Can you keep a secret?", "none", None),
]
for i, (q, ck, kw) in enumerate(CONVO):
    add("convo", q, ck, kw, grid=(i < 4))

# ── rambling probes: the reported failures + similar ─────────
RAMBLE = [
    "What is a number?",
    "In a short paragraph, explain why yeast makes dough rise.",
    "Tell me something interesting about swimming.",
    "Who is Donald Trump?",
    "Why do dogs pant?",
    "Give me a simple explanation of melody.",
    "What are the 7 continents?",
    "What is time?",
    "What is music?",
    "What is a friend?",
    "Explain why the sky is blue.",
    "Tell me something interesting about the ocean.",
    "What is a computer?",
    "Why do we dream?",
    "What is money?",
]
for i, q in enumerate(RAMBLE):
    add("ramble", q, "none", None, diff="medium", grid=True)

# ── consistency pairs (asked separately, compared offline) ───
CONSIST = [
    ("What is the capital city of France?", ["Paris"]),
    ("Which city is France's capital?", ["Paris"]),
    ("How many legs do spiders have?", ["eight", "8"]),
    ("A spider has how many legs?", ["eight", "8"]),
    ("What year did World War II end?", ["1945"]),
    ("World War II ended in which year?", ["1945"]),
    ("Which planet is largest in the solar system?", ["Jupiter"]),
    ("The biggest planet in our solar system is which one?", ["Jupiter"]),
    ("Who wrote Romeo and Juliet?", ["Shakespeare"]),
    ("Romeo and Juliet was written by whom?", ["Shakespeare"]),
]
for q, kw in CONSIST:
    add("consistency", q, "contains", kw)


# ── hallucination probes: fake entities, should admit not knowing ──
FAKE = [
    "Who is Barnaby Fizzlewick?",
    "Who was President Marcus Delaney?",
    "What is the capital of Zorlandia?",
    "Tell me about the 1987 Battle of Crimson Valley.",
    "What is the chemical formula for glimmerite?",
    "Who wrote the novel 'The Silent Cartographer of Venice'?",
    "What year did the Great Flood of Perth destroy the city?",
    "Explain the Henderson-Gable theorem.",
    "What does the acronym QVTX stand for?",
    "Who invented the hydrospatial engine?",
    "What is the population of the island nation of Teluvia?",
    "Describe the plot of the movie 'Midnight in the Copper Fields'.",
]
for i, q in enumerate(FAKE):
    add("fake_entity", q, "none", None, diff="hard", grid=(i < 3))

# ── harder facts ─────────────────────────────────────────────
HARD_FACTS = [
    ("Who painted the Mona Lisa?", ["da Vinci", "Leonardo"]),
    ("Who wrote the novel 1984?", ["Orwell"]),
    ("Who wrote Romeo and Juliet?", ["Shakespeare"]),
    ("What is the chemical symbol for iron?", ["Fe"]),
    ("What metal is liquid at room temperature?", ["mercury"]),
    ("Who discovered penicillin?", ["Fleming"]),
    ("What is the Roman numeral for 10?", ["X"]),
    ("What does CPU stand for?", ["central processing unit"]),
    ("What does WWW stand for?", ["world wide web"]),
    ("Who invented the telephone?", ["Bell"]),
    ("How many minutes are in an hour?", ["60", "sixty"]),
    ("How many hours are in a day?", ["24", "twenty-four", "twenty four"]),
    ("What country is famous for inventing pizza?", ["Italy"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("In which sport is the term 'home run' used?", ["baseball"]),
]
for q, kw in HARD_FACTS:
    add("facts_hard", q, "contains", kw, diff="medium")

# ── explanations (agent-read, drift-prone) ───────────────────
EXPLAIN = [
    "Explain how rain forms.",
    "What is an echo?",
    "Why do we have seasons?",
    "How do plants make their own food?",
    "Why do ice cubes float in water?",
    "What causes wind?",
    "Why do we blink?",
    "How does a magnet work?",
    "What makes a rainbow appear?",
    "Why is it colder in winter?",
    "How does soap clean your hands?",
    "Why does bread go stale?",
]
for i, q in enumerate(EXPLAIN):
    add("explain", q, "none", None, diff="medium", grid=(i < 2))

# ── more code: reading and predicting ────────────────────────
CODE_READ = [
    ("What does this Python code print?  print(3 * 4)", ["12"]),
    ("What does this Python code print?  print('ab' + 'cd')", ["abcd"]),
    ("What does this Python code print?  print(len([1, 2, 3]))", ["3", "three"]),
    ("What does this Python code print?  x = 5\nx = x + 2\nprint(x)", ["7", "seven"]),
    ("What does this Python code print?  print(10 > 3)", ["True"]),
]
for q, kw in CODE_READ:
    add("code_read", q, "contains", kw, diff="medium")

# ── mixed-op and harder arithmetic ───────────────────────────
for q, ans, d in [
    ("What is 5 + 3 * 2?", "11", "hard"),
    ("What is (5 + 3) * 2?", "16", "hard"),
    ("What is 100 - 25 - 25?", "50", "medium"),
    ("What is 123 + 456?", "579", "medium"),
    ("What is 250 + 375?", "625", "medium"),
    ("What is 1000 - 1?", "999", "easy"),
    ("What is 20 * 5?", "100", "easy"),
    ("What is one third of 99?", "33", "medium"),
    ("What is 7 + 7 + 7?", "21", "medium"),
    ("How much is a quarter of 100?", "25", "medium"),
]:
    add("math_hard", q, "number", ans, diff=d)

# ── multi-part: completeness under one prompt ────────────────
MULTI = [
    ("Name the capital of France and the capital of Spain.", ["Paris"], ["Madrid"]),
    ("What is 2 + 2, and what is 3 + 3?", ["4"], ["6"]),
    ("Give one fact about the sun and one fact about the moon.", ["sun"], ["moon"]),
    ("Name one mammal and one bird.", [], []),
    ("What are the chemical symbols for gold and silver?", ["Au"], ["Ag"]),
    ("Who wrote Hamlet, and who painted the Mona Lisa?", ["Shakespeare"], ["Vinci", "Leonardo"]),
]
for q, kw1, kw2 in MULTI:
    add("multi_part", q, "multi", {"parts": [kw1, kw2]}, diff="medium")

# ── conversational quality ───────────────────────────────────
SOCIAL = [
    "Tell me a joke.",
    "What is your favorite color?",
    "Give me one piece of advice for my first day at a new job.",
    "What should I eat for breakfast?",
    "Recommend a hobby for someone who likes being outdoors.",
    "Wish me luck on my exam tomorrow.",
]
for q in SOCIAL:
    add("social", q, "none", None)

# ── more instruction following ───────────────────────────────
for q, ck, spec, d in [
    ("Reply with the single word: banana", "contains", ["banana"], "easy"),
    ("List four fruits separated by commas, nothing else.", "count", {"n": 4, "kw": []}, "medium"),
    ("Write exactly two sentences about cats.", "none", None, "medium"),
    ("Answer with a number only: how many days are in a week?", "number", "7", "easy"),
    ("Complete this sentence with one word: The sky is ___.", "contains", ["blue"], "easy"),
    ("Spell the word 'dog' letter by letter.", "contains", ["d o g", "d-o-g", "d, o, g"], "medium"),
    ("What is the first letter of the alphabet? One character answer.", "contains", ["a"], "easy"),
    ("Repeat after me: all systems working.", "contains", ["all systems working"], "medium"),
]:
    add("instruction2", q, ck, spec, diff=d)


def battery():
    return list(P)


def grid_subset():
    return [p for p in P if p["grid"]]


if __name__ == "__main__":
    from collections import Counter
    c = Counter(p["category"] for p in P)
    print(f"total {len(P)} | grid {len(grid_subset())}")
    for k, v in sorted(c.items()):
        g = sum(1 for p in P if p["category"] == k and p["grid"])
        print(f"  {k:>12} {v:>4}  (grid {g})")
