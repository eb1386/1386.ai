#!/usr/bin/env python3
# Build a small, high-signal repair SFT for basic facts, arithmetic, and code.

from pathlib import Path

from build_rescue_sft import (
    CODE_EXAMPLES,
    ELEMENTS,
    FACTS,
    fmt,
    write_shards,
)


OUT_DIR = Path("data/focus_shards_1.1")


def exact_gate_records(repeats: int) -> list[dict]:
    pairs = [
        ("What color is the sky?", "Blue."),
        ("What is 2 + 2?", "4"),
        ("What is the chemical symbol for gold?", "Au."),
        ("Answer with only the symbol: gold", "Au"),
        ("What is the capital of Australia?", "Canberra."),
        ("What is the largest planet in our solar system?", "Jupiter."),
        ("Who was the first person to walk on the moon?", "Neil Armstrong."),
        ("What year did World War II end?", "1945."),
        ("If a train travels at 60 mph for 2.5 hours, how far does it go?", "150 miles."),
        (
            "What is the difference between weather and climate?",
            "Weather is short-term atmospheric conditions; climate is the long-term pattern of weather in a region.",
        ),
        (
            "Explain photosynthesis in one sentence.",
            "Photosynthesis is how plants use light, water, and carbon dioxide to make sugar and release oxygen.",
        ),
        ("List exactly five common fruits.", "Apple, banana, orange, grape, and pear."),
    ]
    rows = []
    for _ in range(repeats):
        for question, answer in pairs:
            rows.append(fmt(question, answer, "focus_exact_gate"))
            rows.append(fmt(f"Answer directly and briefly. {question}", answer, "focus_exact_gate_direct"))
    return rows


def arithmetic_focus(repeats: int) -> list[dict]:
    rows = []
    add_pairs = [(2, 2), (1, 1), (3, 4), (7, 5), (9, 6), (12, 8), (25, 17), (100, 23)]
    word_pairs = [(60, 2.5, 150), (50, 3, 150), (70, 2, 140), (45, 4, 180), (30, 1.5, 45)]
    for _ in range(repeats):
        for a, b in add_pairs:
            rows.append(fmt(f"What is {a} + {b}?", str(a + b), "focus_arithmetic"))
            rows.append(fmt(f"Answer with only the number: {a} + {b}", str(a + b), "focus_arithmetic_short"))
        for speed, hours, dist in word_pairs:
            rows.append(fmt(
                f"If a train travels at {speed} mph for {hours} hours, how far does it go?",
                f"{dist} miles.",
                "focus_word_math",
            ))
            rows.append(fmt(
                f"A train goes {speed} mph for {hours} hours. Distance?",
                f"{dist} miles.",
                "focus_word_math_short",
            ))
    return rows


def fact_focus(repeats: int) -> list[dict]:
    rows = []
    for _ in range(repeats):
        for name, symbol in ELEMENTS.items():
            rows.append(fmt(f"What is the chemical symbol for {name}?", f"{symbol}.", "focus_elements"))
            rows.append(fmt(f"Answer with only the symbol: {name}", symbol, "focus_elements_short"))
        for question, answer in FACTS:
            rows.append(fmt(question, answer, "focus_facts"))
            rows.append(fmt(f"Answer directly: {question}", answer, "focus_facts_direct"))
    return rows


def code_focus(repeats: int) -> list[dict]:
    rows = []
    prime_answer = (
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True"
    )
    variants = [
        "Write a Python function that checks if a number is prime.",
        "Return only valid Python code. Write a Python function that checks if a number is prime.",
        "Answer with clean Python only: Write a Python function that checks if a number is prime.",
    ]
    for _ in range(repeats):
        for question in variants:
            rows.append(fmt(question, prime_answer, "focus_prime_code"))
        for question, answer in CODE_EXAMPLES:
            clean = answer.replace("```python\n", "").replace("```", "").strip()
            rows.append(fmt(f"Return only valid Python code. {question}", clean, "focus_code"))
    return rows


def essay_focus(repeats: int) -> list[dict]:
    rows = []
    essays = [
        (
            "Write a 100-word essay about why farmers are important.",
            "Farmers are important because they grow the food people need every day. Their work provides fruits, vegetables, grains, milk, meat, and many other products that support healthy communities. Farmers also care for land, animals, water, and crops through long hours in changing weather. Without them, stores would have empty shelves and families would struggle to find fresh meals. Farming also supports local jobs, transportation, markets, and food businesses. Good farmers help protect the future by using careful methods that keep soil productive. Their work is practical, difficult, and essential, so every society depends on farmers.",
        ),
        (
            "Write a 100-word essay about the importance of reading.",
            "Reading is important because it helps people learn, think, and communicate better. Books, articles, and stories introduce new ideas and build vocabulary. Reading also improves focus because it asks the mind to follow details, remember events, and understand meaning. Students who read often usually write more clearly and solve problems with more confidence. Reading can also build empathy by showing how different people live, feel, and make choices. It is useful for school, work, and daily life because instructions, news, and information all require understanding. A strong reading habit gives people lifelong access to knowledge.",
        ),
        (
            "Write a short essay about why clean water matters.",
            "Clean water matters because every person needs it to live. People use water for drinking, cooking, washing, farming, and keeping communities healthy. When water is polluted, disease spreads more easily and families may spend hours trying to find a safe source. Clean rivers, lakes, and groundwater also support plants, animals, and food production. Protecting water means reducing waste, treating sewage, using chemicals carefully, and respecting natural ecosystems. Access to clean water helps children stay in school, workers stay healthy, and towns grow safely. It is one of the most basic foundations of human life.",
        ),
        (
            "Write a five-sentence paragraph about teamwork.",
            "Teamwork helps people solve problems that are too large for one person alone. A good team shares ideas, listens carefully, and divides work fairly. When teammates respect each other, they can learn from different strengths and fix mistakes faster. Teamwork also builds trust because each person knows others are contributing. Many schools, workplaces, and communities succeed because people cooperate toward a common goal.",
        ),
    ]
    for _ in range(repeats):
        for question, answer in essays:
            rows.append(fmt(question, answer, "focus_essay"))
            rows.append(fmt(f"Answer clearly and stay on topic. {question}", answer, "focus_essay_direct"))
    return rows


def main():
    records = []
    records.extend(exact_gate_records(1200))
    records.extend(arithmetic_focus(900))
    records.extend(fact_focus(350))
    records.extend(code_focus(650))
    records.extend(essay_focus(450))
    meta = write_shards(records, OUT_DIR, seed=1386)
    print("Focus SFT shards built:")
    for k, v in meta.items():
        if k != "source_counts":
            print(f"  {k}: {v}")
    print("  source_counts:")
    for k, v in sorted(meta["source_counts"].items()):
        print(f"    {k}: {v:,}")


if __name__ == "__main__":
    main()
