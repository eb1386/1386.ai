# math benchmark

import argparse
import json

import torch
import torch.nn.functional as F
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint


# Embedded demo problems (AMC-style multiple choice)
MATH_PROBLEMS = [
    {
        "question": "What is the value of 2 + 3 * 4?",
        "choices": ["10", "14", "20", "24"],
        "answer": 1,
    },
    {
        "question": "If x = 5, what is x^2 - 3x + 2?",
        "choices": ["10", "12", "14", "17"],
        "answer": 1,
    },
    {
        "question": "What is the sum of the first 10 positive integers?",
        "choices": ["45", "50", "55", "60"],
        "answer": 2,
    },
    {
        "question": "A triangle has sides of length 3, 4, and 5. What is its area?",
        "choices": ["6", "7.5", "10", "12"],
        "answer": 0,
    },
    {
        "question": "What is 15% of 200?",
        "choices": ["20", "25", "30", "35"],
        "answer": 2,
    },
    {
        "question": "How many prime numbers are less than 20?",
        "choices": ["6", "7", "8", "9"],
        "answer": 2,
    },
    {
        "question": "What is the greatest common divisor of 36 and 48?",
        "choices": ["4", "6", "12", "18"],
        "answer": 2,
    },
    {
        "question": "If f(x) = 2x + 1, what is f(f(3))?",
        "choices": ["13", "15", "17", "19"],
        "answer": 1,
    },
    {
        "question": "What is the value of log base 2 of 64?",
        "choices": ["4", "5", "6", "8"],
        "answer": 2,
    },
    {
        "question": "A circle has radius 7. What is its circumference divided by its diameter?",
        "choices": ["3.14", "pi", "7", "14"],
        "answer": 1,
    },
]


def score_choice(
    model: Transformer,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    choice: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    choice_ids = tokenizer.encode(choice, out_type=int)
    full_ids = prompt_ids + choice_ids

    max_len = model.config.max_seq_len
    if len(full_ids) > max_len:
        full_ids = full_ids[-max_len:]
        prompt_ids = full_ids[: -len(choice_ids)]

    tokens = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(tokens)

    start = len(prompt_ids) - 1
    end = len(full_ids) - 1
    log_probs = F.log_softmax(logits[0, start:end], dim=-1)
    target_ids = torch.tensor(choice_ids, device=device)
    scores = log_probs[range(len(choice_ids)), target_ids]
    return scores.sum().item()


def main():
    parser = argparse.ArgumentParser(description="AMC-style math benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/tiny.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    data_cfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(data_cfg["tokenizer_path"])

    correct = 0
    total = len(MATH_PROBLEMS)

    print(f"Running AMC-style math benchmark ({total} problems)...\n")

    for i, prob in enumerate(MATH_PROBLEMS):
        prompt = f"Question: {prob['question']}\nAnswer:"
        scores = []
        for choice in prob["choices"]:
            s = score_choice(model, tokenizer, prompt, f" {choice}", device)
            scores.append(s)

        predicted = max(range(len(scores)), key=lambda j: scores[j])
        is_correct = predicted == prob["answer"]
        correct += int(is_correct)

        mark = "+" if is_correct else "x"
        print(
            f"  [{mark}] Q{i+1}: {prob['question']}"
            f"\n       Predicted: {prob['choices'][predicted]}"
            f"  |  Correct: {prob['choices'][prob['answer']]}"
        )

    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Random baseline: 25.0%")


if __name__ == "__main__":
    main()
