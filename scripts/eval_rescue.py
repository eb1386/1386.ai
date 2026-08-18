#!/usr/bin/env python3
# strict behavioral gate for Plasma checkpoints

import argparse
import json
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


PROMPTS = [
    ("sky", "What color is the sky?", ["blue"], "The sky is blue."),
    ("2plus2", "What is 2 + 2?", ["4"], "4"),
    ("gold", "What is the chemical symbol for gold?", ["Au"], "Au."),
    ("australia", "What is the capital of Australia?", ["Canberra"], "Canberra."),
    ("jupiter", "What is the largest planet in our solar system?", ["Jupiter"], "Jupiter."),
    ("moon", "Who was the first person to walk on the moon?", ["Neil Armstrong"], "Neil Armstrong."),
    ("ww2", "What year did World War II end?", ["1945"], "1945."),
    ("train", "If a train travels at 60 mph for 2.5 hours, how far does it go?", ["150"], "150 miles."),
    ("weather", "What is the difference between weather and climate?", ["short-term", "long-term"], None),
    ("photosynthesis", "Explain photosynthesis in one sentence.", ["plants", "light", "energy"], None),
    ("primecode", "Write a Python function that checks if a number is prime.", ["def", "%", "return False", "return True"], None),
    ("fruits", "List exactly five common fruits.", ["apple", "banana"], None),
    ("farmeressay", "Write a 100-word essay about why farmers are important.", ["farmers", "food", "important"], None),
]

BAD_MARKERS = [
    "as an ai",
    "language model",
    "user:",
    "assistant:",
    "step 6",
    "step 7",
    "step 8",
    "vast, vast",
    "the answer is: 3",
    "the answer is: 9",
]


def clean_response(text: str) -> str:
    text = text.strip()
    for stop in ["\nUser:", "\nAssistant:", "\nSystem:", "\nHuman:", "\nQuestion:", "\n\n\n"]:
        if stop in text:
            text = text[: text.index(stop)]
    return text.strip()


def score_response(pid: str, response: str, expected: list[str]) -> tuple[int, str]:
    low = response.lower()
    if not response:
        return 0, "empty"
    if any(marker in low for marker in BAD_MARKERS):
        return 0, "bad_marker"
    if pid not in {"primecode", "fruits", "farmeressay"} and len(response.split()) > 80:
        return 0, "too_long"

    if pid == "2plus2":
        return (2, "ok") if re.search(r"\b4\b", response) and not re.search(r"\b(3|6|8|9|10)\b", response[:120]) else (0, "wrong_math")
    if pid == "gold":
        return (2, "ok") if re.search(r"\bAu\b", response) else (0, "missing_Au")
    if pid == "australia":
        return (2, "ok") if "canberra" in low else (0, "wrong_capital")
    if pid == "jupiter":
        return (2, "ok") if "jupiter" in low and "sun" not in low[:80] else (0, "wrong_planet")
    if pid == "train":
        return (2, "ok") if "150" in low else (0, "wrong_distance")
    if pid == "primecode":
        ok = all(x in response for x in ["def", "%", "return False", "return True"])
        return (2, "ok") if ok else (0, "broken_code")
    if pid == "fruits":
        fruits = ["apple", "banana", "orange", "grape", "pear", "strawberry", "peach", "mango"]
        hits = sum(1 for fruit in fruits if fruit in low)
        return (2, "ok") if hits >= 5 and len(response.split()) <= 45 else (0, "bad_list")
    if pid == "farmeressay":
        words = response.split()
        hits = sum(1 for exp in expected if exp.lower() in low)
        if hits >= 2 and 75 <= len(words) <= 130:
            return 2, "ok"
        if hits >= 2 and 45 <= len(words) <= 160:
            return 1, "length_off"
        return 0, "bad_essay"

    hits = sum(1 for exp in expected if exp.lower() in low)
    if hits >= min(2, len(expected)):
        return 2, "ok"
    if hits == 1:
        return 1, "partial"
    return 0, "missing_expected"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/finetune_1.1_rescue.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    device = torch.device(args.device)

    model = Transformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(cfg["data"]["tokenizer_path"])

    results = []
    total = 0
    for pid, prompt, expected, ideal in PROMPTS:
        full_prompt = f"User: {prompt}\nAssistant: "
        output = generate(
            model,
            tokenizer,
            full_prompt,
            max_tokens=120,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.4,
            device=device,
        )
        response = clean_response(output[len(full_prompt):])
        score, reason = score_response(pid, response, expected)
        total += score
        results.append({
            "id": pid,
            "prompt": prompt,
            "response": response,
            "expected": expected,
            "ideal": ideal,
            "score": score,
            "reason": reason,
        })
        preview = response.replace("\n", " ")[:180]
        preview = preview.encode("ascii", errors="replace").decode("ascii")
        print(f"[{score}/2] {pid:14s} {reason:16s} {preview}")

    max_score = len(PROMPTS) * 2
    pass_gate = total >= 20 and all(r["score"] > 0 for r in results[:8])
    summary = {"checkpoint": args.checkpoint, "score": total, "max": max_score, "pass": pass_gate, "results": results}
    print(f"\nTOTAL: {total}/{max_score}")
    print(f"PASS: {pass_gate}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    sys.exit(0 if pass_gate else 1)


if __name__ == "__main__":
    main()
