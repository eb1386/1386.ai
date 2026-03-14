# code benchmark

import argparse

import torch
import torch.nn.functional as F
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint


# Embedded code completion problems
CODE_PROBLEMS = [
    {
        "prompt": "def add(a, b):\n    return",
        "choices": [" a + b", " a - b", " a * b", " None"],
        "answer": 0,
        "description": "Simple addition function",
    },
    {
        "prompt": "for i in range(10):\n    print(",
        "choices": ["i)", "i, end=' ')", "range(i))", "10)"],
        "answer": 0,
        "description": "Print loop variable",
    },
    {
        "prompt": "def is_even(n):\n    return n %",
        "choices": [" 2 == 0", " 2 == 1", " 3 == 0", " n == 0"],
        "answer": 0,
        "description": "Even number check",
    },
    {
        "prompt": "numbers = [1, 2, 3, 4, 5]\ntotal = sum(",
        "choices": ["numbers)", "range(5))", "[1,2,3])", "total)"],
        "answer": 0,
        "description": "Sum a list",
    },
    {
        "prompt": "if x > 0:\n    sign = 'positive'\nelse:\n    sign =",
        "choices": [" 'negative'", " 'positive'", " 0", " x"],
        "answer": 0,
        "description": "Sign determination",
    },
    {
        "prompt": "import math\nresult = math.sqrt(",
        "choices": ["16)", "math)", "result)", "'16')"],
        "answer": 0,
        "description": "Square root call",
    },
    {
        "prompt": "def greet(name):\n    return f'Hello, {",
        "choices": ["name}'", "world}'", "greet}'", "None}'"],
        "answer": 0,
        "description": "F-string greeting",
    },
    {
        "prompt": "my_list = []\nmy_list.append(",
        "choices": ["42)", "my_list)", "[])", "()"],
        "answer": 0,
        "description": "Append to list",
    },
    {
        "prompt": "with open('file.txt', 'r') as f:\n    content = f.",
        "choices": ["read()", "write()", "close()", "name"],
        "answer": 0,
        "description": "Read file content",
    },
    {
        "prompt": "class Dog:\n    def __init__(self, name):\n        self.name =",
        "choices": [" name", " 'Dog'", " self", " None"],
        "answer": 0,
        "description": "Constructor assignment",
    },
]


def score_completion(
    model: Transformer,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    completion: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    completion_ids = tokenizer.encode(completion, out_type=int)
    full_ids = prompt_ids + completion_ids

    max_len = model.config.max_seq_len
    if len(full_ids) > max_len:
        full_ids = full_ids[-max_len:]
        prompt_ids = full_ids[: -len(completion_ids)]

    tokens = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(tokens)

    start = len(prompt_ids) - 1
    end = len(full_ids) - 1
    log_probs = F.log_softmax(logits[0, start:end], dim=-1)
    target_ids = torch.tensor(completion_ids, device=device)
    scores = log_probs[range(len(completion_ids)), target_ids]
    return scores.sum().item()


def main():
    parser = argparse.ArgumentParser(description="Code completion benchmark")
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
    total = len(CODE_PROBLEMS)

    print(f"Running code completion benchmark ({total} problems)...\n")

    for i, prob in enumerate(CODE_PROBLEMS):
        scores = []
        for choice in prob["choices"]:
            s = score_completion(model, tokenizer, prob["prompt"], choice, device)
            scores.append(s)

        predicted = max(range(len(scores)), key=lambda j: scores[j])
        is_correct = predicted == prob["answer"]
        correct += int(is_correct)

        mark = "+" if is_correct else "x"
        print(
            f"  [{mark}] {prob['description']}"
            f"\n       Predicted: {prob['choices'][predicted].strip()}"
            f"  |  Correct: {prob['choices'][prob['answer']].strip()}"
        )

    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Random baseline: 25.0%")


if __name__ == "__main__":
    main()
