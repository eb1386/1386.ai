#!/usr/bin/env python3
# shared benchmark prompts + answer checkers for plasma.
#
# used BOTH by the SFT builders (to EXCLUDE these from training, so the benchmark
# stays honest) and by the benchmark harness (to score generations). anything
# added here must be followed by a rebuild of the SFT shards.
#
# scale note: the fixed sets below plus the randomized math/reasoning generated in
# benchmark_v2.py give ~110 scored items. at n=110 a 10-point gap is comfortably
# outside noise; smaller gaps should be read as ties.

import re

# --- factual recall: broad general knowledge, scored by keyword presence ---
FACTUAL = [
    # geography
    {"q": "What is the capital of France?", "kw": ["paris"]},
    {"q": "What is the capital of Japan?", "kw": ["tokyo"]},
    {"q": "What is the capital of Italy?", "kw": ["rome"]},
    {"q": "What is the capital of Canada?", "kw": ["ottawa"]},
    {"q": "What is the largest ocean on Earth?", "kw": ["pacific"]},
    {"q": "What is the longest river in the world?", "kw": ["nile", "amazon"]},
    {"q": "What is the tallest mountain in the world?", "kw": ["everest"]},
    {"q": "In which country are the pyramids of Giza located?", "kw": ["egypt"]},
    {"q": "Which continent is the Sahara Desert in?", "kw": ["africa"]},
    {"q": "What language is primarily spoken in Brazil?", "kw": ["portuguese"]},
    # science
    {"q": "What is the chemical symbol for oxygen?", "kw": ["o2", " o ", "is o", ": o"]},
    {"q": "What is the chemical symbol for water?", "kw": ["h2o", "h₂o"]},
    {"q": "What planet is known as the Red Planet?", "kw": ["mars"]},
    {"q": "Which planet is closest to the Sun?", "kw": ["mercury"]},
    {"q": "What is the largest planet in our solar system?", "kw": ["jupiter"]},
    {"q": "What gas do plants absorb from the air for photosynthesis?", "kw": ["carbon dioxide", "co2"]},
    {"q": "What gas do humans need to breathe to survive?", "kw": ["oxygen"]},
    {"q": "What is the freezing point of water in degrees Celsius?", "kw": ["0", "zero"]},
    {"q": "What is the boiling point of water in degrees Celsius?", "kw": ["100"]},
    {"q": "What is the hardest natural substance on Earth?", "kw": ["diamond"]},
    {"q": "What organ in the human body pumps blood?", "kw": ["heart"]},
    {"q": "How many bones are in the adult human body?", "kw": ["206"]},
    {"q": "What force keeps planets in orbit around the Sun?", "kw": ["gravity", "gravitation"]},
    {"q": "What is the center of an atom called?", "kw": ["nucleus"]},
    # history / literature / arts
    {"q": "Who wrote the play Romeo and Juliet?", "kw": ["shakespeare"]},
    {"q": "Who painted the Mona Lisa?", "kw": ["leonardo", "da vinci", "vinci"]},
    {"q": "Who developed the theory of relativity?", "kw": ["einstein"]},
    {"q": "Who was the first person to walk on the Moon?", "kw": ["armstrong"]},
    {"q": "In what year did the Second World War end?", "kw": ["1945"]},
    {"q": "Who was the first President of the United States?", "kw": ["washington"]},
    {"q": "What ancient civilization built the Colosseum?", "kw": ["roman", "rome"]},
    # everyday / common sense
    {"q": "How many continents are there on Earth?", "kw": ["seven", "7"]},
    {"q": "How many days are there in a leap year?", "kw": ["366"]},
    {"q": "How many minutes are there in one hour?", "kw": ["60", "sixty"]},
    {"q": "What is the currency used in the United Kingdom?", "kw": ["pound", "sterling", "gbp"]},
    {"q": "What colour do you get when you mix blue and yellow?", "kw": ["green"]},
    {"q": "How many sides does a triangle have?", "kw": ["3", "three"]},
    {"q": "What season comes after summer?", "kw": ["autumn", "fall"]},
    {"q": "What do bees produce?", "kw": ["honey"]},
    {"q": "What is the opposite of hot?", "kw": ["cold", "cool"]},
]

# --- simple coding: generated function is EXECUTED against hidden tests ---
CODE = [
    {"q": "Write a Python function called add(a, b) that returns the sum of a and b. Only output the function.",
     "fn": "add", "tests": [((2, 3), 5), ((-1, 1), 0), ((10, 20), 30)]},
    {"q": "Write a Python function called reverse_string(s) that returns the string reversed. Only output the function.",
     "fn": "reverse_string", "tests": [(("hello",), "olleh"), (("abc",), "cba"), (("",), "")]},
    {"q": "Write a Python function called is_even(n) that returns True if n is even and False otherwise. Only output the function.",
     "fn": "is_even", "tests": [((4,), True), ((7,), False), ((0,), True)]},
    {"q": "Write a Python function called factorial(n) that returns n factorial. Only output the function.",
     "fn": "factorial", "tests": [((0,), 1), ((1,), 1), ((5,), 120)]},
    {"q": "Write a Python function called max_of_two(a, b) that returns the larger of a and b. Only output the function.",
     "fn": "max_of_two", "tests": [((3, 9), 9), ((10, 2), 10), ((-5, -8), -5)]},
    {"q": "Write a Python function called count_vowels(s) that returns the number of vowels (a, e, i, o, u) in s. Only output the function.",
     "fn": "count_vowels", "tests": [(("hello",), 2), (("sky",), 0), (("aeiou",), 5)]},
    {"q": "Write a Python function called square(n) that returns n multiplied by itself. Only output the function.",
     "fn": "square", "tests": [((3,), 9), ((0,), 0), ((-4,), 16)]},
    {"q": "Write a Python function called sum_list(nums) that returns the sum of a list of numbers. Only output the function.",
     "fn": "sum_list", "tests": [(([1, 2, 3],), 6), (([],), 0), (([-1, 1],), 0)]},
    {"q": "Write a Python function called is_palindrome(s) that returns True if s reads the same forwards and backwards. Only output the function.",
     "fn": "is_palindrome", "tests": [(("racecar",), True), (("hello",), False), (("aa",), True)]},
    {"q": "Write a Python function called celsius_to_fahrenheit(c) that converts Celsius to Fahrenheit. Only output the function.",
     "fn": "celsius_to_fahrenheit", "tests": [((0,), 32), ((100,), 212), ((-40,), -40)]},
    {"q": "Write a Python function called longest_word(words) that returns the longest string in a list. Only output the function.",
     "fn": "longest_word", "tests": [((["a", "bbb", "cc"],), "bbb"), ((["x"],), "x")]},
    {"q": "Write a Python function called double_all(nums) that returns a list with every number doubled. Only output the function.",
     "fn": "double_all", "tests": [(([1, 2, 3],), [2, 4, 6]), (([],), []), (([-1],), [-2])]},
]

# --- instruction following: produce exactly N distinct items ---
INSTRUCTION = [
    {"q": "List exactly three primary colors.", "n": 3},
    {"q": "Name four different fruits.", "n": 4},
    {"q": "List three planets in our solar system.", "n": 3},
    {"q": "Name two oceans on Earth.", "n": 2},
    {"q": "List three days of the week.", "n": 3},
    {"q": "Name three animals that live in the ocean.", "n": 3},
    {"q": "List four colors.", "n": 4},
    {"q": "Name three countries in Europe.", "n": 3},
    {"q": "List three things you might find in a kitchen.", "n": 3},
    {"q": "Name two musical instruments.", "n": 2},
]

# --- short writing: length + on-topic + non-repetitive + non-refusal ---
ESSAY = [
    {"q": "Write a short paragraph explaining why regular exercise is good for health.",
     "kw": ["exercise", "health", "body", "heart", "muscle", "fit", "strong"]},
    {"q": "Explain in two or three sentences what photosynthesis is.",
     "kw": ["plant", "light", "sun", "energy", "carbon", "oxygen", "glucose"]},
    {"q": "Write a few sentences about why reading books is beneficial.",
     "kw": ["read", "book", "knowledge", "learn", "vocabulary", "imagination", "mind"]},
    {"q": "Briefly describe what the water cycle is.",
     "kw": ["water", "evaporat", "rain", "cloud", "precipitat", "condens", "cycle"]},
    {"q": "Explain in a short paragraph why sleep is important for people.",
     "kw": ["sleep", "rest", "brain", "body", "health", "energy", "memory"]},
    {"q": "Write a short paragraph about why recycling matters.",
     "kw": ["recycl", "waste", "environment", "plastic", "reuse", "planet", "pollut"]},
    {"q": "Explain briefly what the internet is used for.",
     "kw": ["internet", "information", "communicat", "website", "online", "connect", "network"]},
    {"q": "Write a few sentences describing the seasons of the year.",
     "kw": ["winter", "spring", "summer", "autumn", "fall", "season", "weather"]},
]

# --- conversational: coherent + non-refusal ---
CONVO = [
    "Hello! How are you today?",
    "Can you help me with something?",
    "What kind of questions can you answer?",
    "Tell me a little about yourself.",
    "Good morning!",
    "Thanks for your help.",
]


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def all_prompt_strings() -> set:
    """normalized set of every fixed benchmark user-prompt, for SFT decontamination."""
    out = set()
    for grp in (FACTUAL, CODE, INSTRUCTION, ESSAY):
        for item in grp:
            out.add(_norm(item["q"]))
    for q in CONVO:
        out.add(_norm(q))
    return out


def norm_prompt(s: str) -> str:
    return _norm(s)
