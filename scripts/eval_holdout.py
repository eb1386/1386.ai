#!/usr/bin/env python3
# Holdout behavior eval: variant prompts not identical to focused repair gates.

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


TESTS = [
    ("iron", "What is the chemical symbol for iron?", lambda r: re.search(r"\bFe\b", r) is not None),
    ("oxygen", "Give only the chemical symbol for oxygen.", lambda r: re.fullmatch(r"\s*O\.?\s*", r) is not None),
    ("silver", "Silver's element symbol is what?", lambda r: re.search(r"\bAg\b", r) is not None),
    ("add17", "What is 17 + 29?", lambda r: re.search(r"\b46\b", r) is not None),
    ("subtract", "What is 81 - 37?", lambda r: re.search(r"\b44\b", r) is not None),
    ("multiply", "What is 13 times 9?", lambda r: re.search(r"\b117\b", r) is not None),
    ("bus", "A bus travels at 40 mph for 3.5 hours. How far does it travel?", lambda r: re.search(r"\b140\b", r) is not None),
    ("bike", "A bike moves 12 miles per hour for 2 hours. What distance does it cover?", lambda r: re.search(r"\b24\b", r) is not None),
    ("canada", "What is the capital of Canada?", lambda r: "ottawa" in r.lower()),
    ("mars", "Which planet is called the Red Planet?", lambda r: "mars" in r.lower()),
    ("water", "What is H2O commonly called?", lambda r: "water" in r.lower()),
    ("prime_variant", "Create a Python function named check_prime that returns True for prime numbers.", lambda r: all(x in r for x in ["def", "%", "return False", "return True"])),
    ("sum_code", "Return only Python code for a function that adds two numbers.", lambda r: "def" in r and "+" in r and "return" in r),
    ("farmer_essay", "Write a 100-word essay about why farmers matter.", lambda r: 70 <= len(r.split()) <= 135 and "farm" in r.lower() and "food" in r.lower()),
    ("teacher_essay", "Write a short essay about why teachers are important.", lambda r: 45 <= len(r.split()) <= 140 and "teacher" in r.lower() and ("learn" in r.lower() or "student" in r.lower())),
    ("complete", "Complete the sentence: Bob was a farmer who", lambda r: 8 <= len(r.split()) <= 60 and "farm" in r.lower()),
]


def clean_response(text: str) -> str:
    text = text.strip()
    for stop in ["\nUser:", "\nAssistant:", "\nSystem:", "\nHuman:", "\nQuestion:", "\n\n\n"]:
        if stop in text:
            text = text[: text.index(stop)]
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/finetune_1.1_focus.yaml")
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
    passed = 0
    for test_id, prompt, check in TESTS:
        full_prompt = f"User: {prompt}\nAssistant: "
        output = generate(
            model,
            tokenizer,
            full_prompt,
            max_tokens=140,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.35,
            device=device,
        )
        response = clean_response(output[len(full_prompt):])
        ok = bool(check(response))
        passed += int(ok)
        preview = response.replace("\n", " ")[:180].encode("ascii", errors="replace").decode("ascii")
        print(f"[{'PASS' if ok else 'FAIL'}] {test_id:14s} {preview}")
        results.append({"id": test_id, "prompt": prompt, "response": response, "pass": ok})

    summary = {"checkpoint": args.checkpoint, "passed": passed, "total": len(TESTS), "results": results}
    print(f"\nTOTAL: {passed}/{len(TESTS)}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    sys.exit(0 if passed >= 13 else 1)


if __name__ == "__main__":
    main()
