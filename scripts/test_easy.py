# very easy single-word completions

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint
from src.inference.generate import generate


PROMPTS = [
    ("The country east of Poland and west of Russia is", "Ukraine"),
    ("The Statue of Liberty stands in the city of New", "York"),
    ("The Vatican is in the city of", "Rome"),
    ("The Sahara is the largest", "desert"),
    ("Sushi is a traditional dish from the country of", "Japan"),
    ("Pizza originated in the country of", "Italy"),
    ("The Beatles were a band from the country of", "England"),
    ("The Amazon rainforest is mostly in the country of", "Brazil"),
    ("Pyramids are most famously located in", "Egypt"),
    ("Wine is made from", "grapes"),
    ("Apples grow on", "trees"),
    ("Fish live in", "water"),
    ("Most birds can", "fly"),
    ("Cats say meow, dogs say", "woof/bark"),
    ("The sky during the day is", "blue"),
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

    correct = 0
    for i, (p, expected) in enumerate(PROMPTS, 1):
        out = generate(
            model, tok, p,
            max_tokens=10,
            temperature=0.2,
            top_k=10,
            top_p=0.85,
            repetition_penalty=1.2,
            device=device,
        )
        new = out[len(p):].strip().split("\n")[0].split(".")[0].split(",")[0]
        new = new.strip()
        is_match = expected.lower().split("/")[0] in new.lower() or any(
            e.lower() in new.lower() for e in expected.split("/")
        )
        marker = "✓" if is_match else "✗"
        if is_match:
            correct += 1
        print(f"[{i:2d}] {marker} {p}")
        print(f"       got:      '{new}'")
        print(f"       expected: '{expected}'")

    print(f"\nScore: {correct}/{len(PROMPTS)} = {100*correct/len(PROMPTS):.0f}%")


if __name__ == "__main__":
    main()
