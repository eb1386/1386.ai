#!/usr/bin/env python3
# Balanced SFT: improve instruction following and writing without cramming facts.

import argparse
import random
from pathlib import Path

from build_rescue_sft import CODE_EXAMPLES, fmt, iter_public_records, write_shards


OUT_DIR = Path("data/balanced_shards_1.1")


def essay_records(repeats: int) -> list[dict]:
    rows = []
    samples = [
        (
            "Write a 100-word essay about why farmers are important.",
            "Farmers are important because they grow the food people need every day. Their work provides fruits, vegetables, grains, milk, meat, and many other products that support healthy communities. Farmers also care for land, animals, water, and crops through long hours in changing weather. Without them, stores would have empty shelves and families would struggle to find fresh meals. Farming supports local jobs, transportation, markets, and food businesses. Good farmers protect the future by using careful methods that keep soil productive. Their work is practical, difficult, and essential, so every society depends on farmers.",
        ),
        (
            "Write a 100-word essay about why teachers are important.",
            "Teachers are important because they help students build knowledge, confidence, and useful skills. A good teacher explains difficult ideas clearly, notices when students are confused, and encourages steady effort. Teachers also help young people practice reading, writing, math, science, and problem solving. Beyond facts, they model patience, fairness, and curiosity. Their work supports families and communities because educated students can make better choices and prepare for future jobs. Many people remember a teacher who helped them believe they could improve. For these reasons, teachers play a major role in every strong society.",
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
        for prompt, answer in samples:
            rows.append(fmt(prompt, answer, "balanced_essay"))
            rows.append(fmt(f"Answer clearly and stay on topic. {prompt}", answer, "balanced_essay_direct"))
    return rows


def concise_behavior_records(repeats: int) -> list[dict]:
    pairs = [
        ("What color is the sky?", "Blue."),
        ("List exactly five common fruits.", "Apple, banana, orange, grape, and pear."),
        ("Explain photosynthesis in one sentence.", "Photosynthesis is how plants use light, water, and carbon dioxide to make sugar and release oxygen."),
        ("What is the difference between weather and climate?", "Weather is short-term atmospheric conditions; climate is the long-term pattern of weather in a region."),
        ("Complete the sentence: Bob was a farmer who", "Bob was a farmer who worked hard each morning, cared for his crops, and helped feed his town."),
    ]
    rows = []
    for _ in range(repeats):
        for prompt, answer in pairs:
            rows.append(fmt(prompt, answer, "balanced_concise"))
    return rows


def procedural_math_records(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.randint(0, 200)
        b = rng.randint(0, 200)
        rows.append(fmt(f"What is {a} + {b}?", str(a + b), "balanced_math"))
        x = rng.randint(0, 200)
        y = rng.randint(0, x)
        rows.append(fmt(f"What is {x} - {y}?", str(x - y), "balanced_math"))
        m = rng.randint(2, 20)
        k = rng.randint(2, 20)
        rows.append(fmt(f"What is {m} times {k}?", str(m * k), "balanced_math"))

    for _ in range(250):
        speed = rng.choice([20, 30, 40, 45, 50, 55, 60, 65, 70])
        hours = rng.choice([1, 1.5, 2, 2.5, 3, 3.5, 4])
        dist = speed * hours
        dist_text = str(int(dist)) if float(dist).is_integer() else str(dist)
        vehicle = rng.choice(["train", "bus", "car", "bike"])
        rows.append(fmt(
            f"A {vehicle} travels at {speed} mph for {hours} hours. How far does it travel?",
            f"{dist_text} miles.",
            "balanced_word_math",
        ))
    return rows


def code_records(repeats: int) -> list[dict]:
    rows = []
    for _ in range(repeats):
        for question, answer in CODE_EXAMPLES:
            clean = answer.replace("```python\n", "").replace("```", "").strip()
            rows.append(fmt(question, clean, "balanced_code"))
            rows.append(fmt(f"Return only valid Python code. {question}", clean, "balanced_code"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=138611)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and (out_dir / "meta.yaml").exists() and not args.force:
        print(f"[skip] existing balanced shards: {out_dir}")
        return
    if out_dir.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite incomplete existing directory without --force: {out_dir}")

    records = []
    records.extend(essay_records(500))
    records.extend(concise_behavior_records(600))
    records.extend(procedural_math_records(seed=args.seed, n=3500))
    records.extend(code_records(350))

    limits = {
        "instruct_no_robots": 12_000,
        "instruct_capybara": 8_000,
        "instruct_metamath": 12_000,
        "instruct_wizardlm": 8_000,
        "instruct_ultrachat": 12_000,
        "instruct_slimorca": 12_000,
        "synthetic_instruct": 6_000,
    }
    records.extend(iter_public_records(seed=args.seed, limits=limits))

    meta = write_shards(records, out_dir, seed=args.seed)
    print("Balanced SFT shards built:")
    for k, v in meta.items():
        if k != "source_counts":
            print(f"  {k}: {v}")
    print("  source_counts:")
    for k, v in sorted(meta["source_counts"].items()):
        print(f"    {k}: {v:,}")


if __name__ == "__main__":
    main()
