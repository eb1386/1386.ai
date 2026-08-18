# very high-frequency bigram completions

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
    ("Mary had a little", "lamb"),
    ("Twinkle twinkle little", "star"),
    ("Once upon a", "time"),
    ("Read between the", "lines"),
    ("United States of", "America"),
    ("World Wide", "Web"),
    ("The Lord of the", "Rings"),
    ("Trick or", "treat"),
    ("Salt and", "pepper"),
    ("Bread and", "butter"),
    ("Black and", "white"),
    ("Rock and", "roll"),
    ("Peanut butter and", "jelly"),
    ("Hide and", "seek"),
    ("Cat and", "mouse"),
    ("Macaroni and", "cheese"),
    ("Knife and", "fork"),
    ("Husband and", "wife"),
    ("Brother and", "sister"),
    ("Heads and", "tails"),
    ("Up and", "down"),
    ("Yin and", "yang"),
    ("Day and", "night"),
    ("Bacon and", "eggs"),
    ("Hugs and", "kisses"),
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
            max_tokens=4,
            temperature=0.1,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.0,
            device=device,
        )
        new = out[len(p):].strip().split("\n")[0]
        new_first = new.split()[0] if new.split() else new
        new_first_clean = new_first.rstrip(",.!?;:").lower()
        is_match = expected.lower() == new_first_clean or expected.lower() in new.lower()[:20]
        marker = "✓" if is_match else "✗"
        if is_match:
            correct += 1
        print(f"[{i:2d}] {marker} '{p}' -> '{new[:30]}' (expected: {expected})")

    print(f"\nScore: {correct}/{len(PROMPTS)} = {100*correct/len(PROMPTS):.0f}%")


if __name__ == "__main__":
    main()
