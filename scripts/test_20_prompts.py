# 20-prompt diverse completion test on mid-pretrain checkpoint

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint
from src.inference.generate import generate


PROMPTS = [
    # geography
    "The Eiffel Tower is located in the city of",
    "The Pacific Ocean is the world's",
    # science
    "Water boils at a temperature of",
    "The chemical symbol for gold is",
    "DNA stands for",
    # history
    "World War II ended in the year",
    "The first man to walk on the moon was",
    # language / definitions
    "A synonym for 'happy' is",
    "The opposite of 'big' is",
    "A bird that cannot fly is the",
    # common sense
    "Before you can drive a car, you need to",
    "If you mix red and blue paint, you get",
    # patterns
    "1, 2, 3, 4, 5,",
    "Monday, Tuesday, Wednesday,",
    # narrative continuation
    "She opened the door and saw",
    "The detective examined the clues and concluded that",
    # code-ish
    "def add(a, b):\n    return",
    "for i in range(10):\n    print(",
    # famous quotes / completions
    "To be or not to be, that is the",
    "Roses are red, violets are",
]


def main():
    ckpt = "checkpoints/1.1_step_120000.pt"
    cfg = load_config("configs/pretrain_1.1.yaml")
    model_cfg = ModelConfig.from_dict(cfg["model"])
    device = torch.device("cpu")

    print(f"loading {ckpt}...")
    model = Transformer(model_cfg).to(device)
    load_checkpoint(ckpt, model)
    model.eval()
    print(f"  {model.count_parameters():,} params\n")

    tok = spm.SentencePieceProcessor()
    tok.load(cfg["data"]["tokenizer_path"])

    for i, p in enumerate(PROMPTS, 1):
        print(f"\n[{i:2d}] >>> {p}")
        t0 = time.time()
        out = generate(
            model, tok, p,
            max_tokens=30,
            temperature=0.4,
            top_k=20,
            top_p=0.85,
            repetition_penalty=1.2,
            device=device,
        )
        new = out[len(p):].strip()
        # truncate at first newline-newline for readability
        new = new.split("\n\n")[0]
        dt = time.time() - t0
        print(f"     {new}")
        print(f"     [{dt:.1f}s]")


if __name__ == "__main__":
    main()
