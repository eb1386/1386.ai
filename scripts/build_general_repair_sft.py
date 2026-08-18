#!/usr/bin/env python3
# Broad repair SFT for generalization, not exact prompt memorization.

import random
from pathlib import Path

from build_rescue_sft import CAPITALS, CODE_EXAMPLES, ELEMENTS, FACTS, fmt, write_shards


OUT_DIR = Path("data/general_repair_shards_1.1")


def element_records(repeats: int) -> list[dict]:
    rows = []
    templates = [
        "What is the chemical symbol for {name}?",
        "Give only the chemical symbol for {name}.",
        "{name}'s element symbol is what?",
        "In chemistry, what symbol represents {name}?",
        "Answer briefly: symbol for {name}.",
    ]
    for _ in range(repeats):
        for name, symbol in ELEMENTS.items():
            for template in templates:
                answer = symbol if "only" in template or "briefly" in template else f"{symbol}."
                rows.append(fmt(template.format(name=name), answer, "general_elements"))
    return rows


def fact_records(repeats: int) -> list[dict]:
    rows = []
    for _ in range(repeats):
        for country, capital in CAPITALS.items():
            rows.append(fmt(f"What is the capital of {country}?", f"{capital}.", "general_capitals"))
            rows.append(fmt(f"Name {country}'s capital city.", f"{capital}.", "general_capitals"))
        for question, answer in FACTS:
            rows.append(fmt(question, answer, "general_facts"))
            rows.append(fmt(f"Answer in a few words: {question}", answer, "general_facts"))
    return rows


def math_records(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.randint(0, 250)
        b = rng.randint(0, 250)
        rows.append(fmt(f"What is {a} + {b}?", str(a + b), "general_math"))
        rows.append(fmt(f"Answer with only the number: {a} plus {b}", str(a + b), "general_math"))
        x = rng.randint(0, 250)
        y = rng.randint(0, x)
        rows.append(fmt(f"What is {x} - {y}?", str(x - y), "general_math"))
        m = rng.randint(2, 20)
        k = rng.randint(2, 20)
        rows.append(fmt(f"What is {m} times {k}?", str(m * k), "general_math"))

    vehicles = ["train", "bus", "car", "bike", "boat"]
    speeds = [12, 20, 30, 40, 45, 50, 55, 60, 65, 70, 80]
    hours = [0.5, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4]
    for _ in range(250):
        for vehicle in vehicles:
            speed = rng.choice(speeds)
            hour = rng.choice(hours)
            dist = speed * hour
            dist_text = str(int(dist)) if float(dist).is_integer() else str(dist)
            rows.append(fmt(
                f"A {vehicle} travels at {speed} mph for {hour} hours. How far does it travel?",
                f"{dist_text} miles.",
                "general_word_math",
            ))
            rows.append(fmt(
                f"If a {vehicle} goes {speed} miles per hour for {hour} hours, what distance does it cover?",
                f"{dist_text} miles.",
                "general_word_math",
            ))
    return rows


def code_records(repeats: int) -> list[dict]:
    rows = []
    prime_answers = [
        (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True"
        ),
        (
            "def check_prime(number):\n"
            "    if number < 2:\n"
            "        return False\n"
            "    divisor = 2\n"
            "    while divisor * divisor <= number:\n"
            "        if number % divisor == 0:\n"
            "            return False\n"
            "        divisor += 1\n"
            "    return True"
        ),
    ]
    prime_prompts = [
        "Write a Python function that checks if a number is prime.",
        "Create a Python function named check_prime that returns True for prime numbers.",
        "Return only Python code for a prime-number checker.",
    ]
    for _ in range(repeats):
        for prompt in prime_prompts:
            for answer in prime_answers:
                rows.append(fmt(prompt, answer, "general_prime_code"))
        for question, answer in CODE_EXAMPLES:
            clean = answer.replace("```python\n", "").replace("```", "").strip()
            rows.append(fmt(question, clean, "general_code"))
            rows.append(fmt(f"Return only valid Python code. {question}", clean, "general_code"))
    return rows


def essay_records(repeats: int) -> list[dict]:
    topics = [
        ("farmers", "farmers", "food", "communities"),
        ("teachers", "teachers", "students", "learning"),
        ("reading", "reading", "knowledge", "thinking"),
        ("clean water", "clean water", "health", "communities"),
        ("exercise", "exercise", "health", "energy"),
        ("teamwork", "teamwork", "people", "goals"),
        ("sleep", "sleep", "health", "focus"),
        ("libraries", "libraries", "books", "communities"),
    ]
    rows = []
    for _ in range(repeats):
        for topic, key1, key2, key3 in topics:
            essay = (
                f"{key1.capitalize()} are important because they support {key2} and strengthen {key3}. "
                f"They help people meet daily needs, solve problems, and build better habits. "
                f"When a community values {topic}, it becomes healthier, more informed, and more prepared for the future. "
                f"This work often requires patience, responsibility, and steady effort. "
                f"It also connects people because everyone benefits when basic needs are met well. "
                f"For these reasons, {topic} should be respected, protected, and improved by families, schools, and local leaders."
            )
            rows.append(fmt(f"Write a 100-word essay about why {topic} are important.", essay, "general_essay"))
            rows.append(fmt(f"Write a short essay about why {topic} matter.", essay, "general_essay"))
    rows.append(fmt(
        "Complete the sentence: Bob was a farmer who",
        "Bob was a farmer who worked hard every morning, cared for his crops, and helped feed the families in his town.",
        "general_completion",
    ))
    return rows


def main():
    records = []
    records.extend(element_records(180))
    records.extend(fact_records(100))
    records.extend(math_records(seed=20260512, n=6000))
    records.extend(code_records(350))
    records.extend(essay_records(450))
    meta = write_shards(records, OUT_DIR, seed=20260512)
    print("General repair SFT shards built:")
    for k, v in meta.items():
        if k != "source_counts":
            print(f"  {k}: {v}")
    print("  source_counts:")
    for k, v in sorted(meta["source_counts"].items()):
        print(f"    {k}: {v:,}")


if __name__ == "__main__":
    main()
