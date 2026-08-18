# 25 prompts targeted at the base model's known strengths

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
    # repeats of past wins
    ("Once upon a", "time"),
    ("World Wide", "Web"),
    ("United States of", "America"),
    ("The Lord of the", "Rings"),
    ("Roses are red, violets are", "blue"),
    # number sequences
    ("1, 2, 3, 4, 5, 6,", "7"),
    ("10, 20, 30, 40, 50,", "60"),
    ("January, February, March, April, May, June, July, August,", "September"),
    ("Sunday, Monday, Tuesday, Wednesday, Thursday, Friday,", "Saturday"),
    ("A, B, C, D, E, F, G, H, I, J, K, L, M, N,", "O"),
    # titles / named entities
    ("Star Wars: A New", "Hope"),
    ("The Statue of", "Liberty"),
    ("World Health", "Organization"),
    ("Encyclopedia", "Britannica"),
    ("World War", "II"),
    # code
    ('def hello_world():\n    print("', "Hello"),
    ("import nu", "mpy"),
    ("for i in ra", "nge"),
    # wikipedia openers - judge for topical relevance
    ("Albert Einstein was a German-born theoretical", "physicist"),
    ("The Earth is the third planet from the", "Sun"),
    ("Leonardo da Vinci was an Italian polymath of the High", "Renaissance"),
    ("World War II was a global war that lasted from 1939 to", "1945"),
    ("Photosynthesis is the process used by plants to convert sunlight into chemical", "energy"),
    ("DNA is a polymer composed of two polynucleotide chains that coil around each other to form a double", "helix"),
    ("Pi is approximately equal to 3.", "14"),
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
            max_tokens=8,
            temperature=0.1,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.0,
            device=device,
        )
        new = out[len(p):].strip()
        # check first ~30 chars for the expected answer
        is_match = expected.lower() in new.lower()[:40]
        marker = "✓" if is_match else "✗"
        if is_match:
            correct += 1
        print(f"[{i:2d}] {marker} '{p[-40:] if len(p) > 40 else p}' -> '{new[:30]}' (want: {expected})")

    print(f"\nScore: {correct}/{len(PROMPTS)} = {100*correct/len(PROMPTS):.0f}%")


if __name__ == "__main__":
    main()
