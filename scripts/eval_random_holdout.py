#!/usr/bin/env python3
# Randomized holdout probes for SFT checkpoints. This intentionally changes
# prompts per run so exact-prompt memorization does not look like success.

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sentencepiece as spm
import torch

from src.inference.generate import generate
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_checkpoint, load_config


ELEMENTS = {
    "hydrogen": "H", "carbon": "C", "oxygen": "O", "sodium": "Na",
    "iron": "Fe", "copper": "Cu", "silver": "Ag", "gold": "Au",
    "lead": "Pb", "chlorine": "Cl", "potassium": "K", "calcium": "Ca",
}

CAPITALS = {
    "Canada": "Ottawa", "Japan": "Tokyo", "Brazil": "Brasilia",
    "Egypt": "Cairo", "Kenya": "Nairobi", "Italy": "Rome",
    "Spain": "Madrid", "South Korea": "Seoul", "Mexico": "Mexico City",
}

SCIENCE = [
    ("What planet is known as the Red Planet?", ["mars"]),
    ("What gas do plants release during photosynthesis?", ["oxygen"]),
    ("What force pulls objects toward Earth?", ["gravity"]),
    ("What is H2O commonly called?", ["water"]),
    ("What organ pumps blood through the body?", ["heart"]),
]

ESSAY_TOPICS = ["farmers", "teachers", "clean water", "reading", "exercise", "teamwork"]


def clean(text: str) -> str:
    text = text.strip()
    for stop in ["\nUser:", "\nAssistant:", "\nSystem:", "\nHuman:", "\nQuestion:", "\n\n\n"]:
        if stop in text:
            text = text[: text.index(stop)]
    return text.strip()


def make_tests(seed: int, n: int):
    rng = random.Random(seed)
    tests = []

    for _ in range(3):
        name, sym = rng.choice(list(ELEMENTS.items()))
        prompt = rng.choice([
            f"What is the chemical symbol for {name}?",
            f"Give only the element symbol for {name}.",
            f"In chemistry, what symbol represents {name}?",
        ])
        tests.append((f"element_{name}", prompt, lambda r, s=sym: re.search(rf"\b{re.escape(s)}\b", r) is not None))

    for _ in range(3):
        country, capital = rng.choice(list(CAPITALS.items()))
        tests.append((f"capital_{country}", f"What is the capital of {country}?", lambda r, c=capital: c.lower() in r.lower()))

    for _ in range(4):
        a = rng.randint(3, 120)
        b = rng.randint(3, 120)
        tests.append((f"add_{a}_{b}", f"What is {a} + {b}?", lambda r, ans=str(a + b): re.search(rf"\b{ans}\b", r) is not None))

    for _ in range(3):
        speed = rng.choice([12, 25, 30, 40, 45, 50, 60, 70])
        hours = rng.choice([1.5, 2, 2.5, 3, 3.5, 4])
        ans = speed * hours
        ans_text = str(int(ans)) if float(ans).is_integer() else str(ans)
        vehicle = rng.choice(["train", "bus", "bike", "car"])
        tests.append((f"distance_{speed}_{hours}", f"A {vehicle} travels {speed} mph for {hours} hours. How far does it go?", lambda r, a=ans_text: re.search(rf"\b{re.escape(a)}\b", r) is not None))

    for q, expected in rng.sample(SCIENCE, k=3):
        tests.append(("science", q, lambda r, exp=expected: any(x in r.lower() for x in exp)))

    topic = rng.choice(ESSAY_TOPICS)
    tests.append((f"essay_{topic}", f"Write a 100-word essay about why {topic} are important.", lambda r, t=topic: 70 <= len(r.split()) <= 140 and t.split()[0].lower() in r.lower()))

    tests.append(("prime_code", "Return only Python code for a function that checks whether n is prime.", lambda r: all(x in r for x in ["def", "%", "return False", "return True"])))

    rng.shuffle(tests)
    return tests[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(1_000_000_000)
    cfg = load_config(args.config)
    model = Transformer(ModelConfig.from_dict(cfg["model"])).to(torch.device(args.device))
    load_checkpoint(args.checkpoint, model)
    model.eval()
    tok = spm.SentencePieceProcessor()
    tok.load(cfg["data"]["tokenizer_path"])

    results = []
    passed = 0
    for tid, prompt, check in make_tests(seed, args.n):
        full = f"User: {prompt}\nAssistant: "
        out = generate(model, tok, full, max_tokens=150, temperature=0.0, top_k=1, top_p=1.0, repetition_penalty=1.35, device=torch.device(args.device))
        response = clean(out[len(full):])
        ok = bool(check(response))
        passed += int(ok)
        preview = response.replace("\n", " ")[:160].encode("ascii", errors="replace").decode("ascii")
        print(f"[{'PASS' if ok else 'FAIL'}] {tid:22s} {preview}")
        results.append({"id": tid, "prompt": prompt, "response": response, "pass": ok})

    summary = {"checkpoint": args.checkpoint, "seed": seed, "passed": passed, "total": len(results), "results": results}
    print(f"\nTOTAL: {passed}/{len(results)} seed={seed}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    sys.exit(0 if passed >= max(10, int(0.7 * len(results))) else 1)


if __name__ == "__main__":
    main()
