# 20 prompts, 8 easy + 12 wikipedia-style continuations

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
    # 8 easy
    ("Once upon a", "time"),
    ("1, 2, 3, 4, 5,", "6"),
    ("Sunday, Monday, Tuesday, Wednesday, Thursday, Friday,", "Saturday"),
    ("United States of", "America"),
    ("World War", "II"),
    ("World Wide", "Web"),
    ("The Lord of the", "Rings"),
    ("Roses are red, violets are", "blue"),
    # 12 wikipedia-style
    ("Albert Einstein was a German-born theoretical", "physicist"),
    ("Leonardo da Vinci was an Italian polymath of the High", "Renaissance"),
    ("World War II was a global war that lasted from 1939 to", "1945"),
    ("Photosynthesis is the process used by plants to convert sunlight into chemical", "energy"),
    ("DNA is a polymer composed of two polynucleotide chains that coil around each other to form a double", "helix"),
    ("The Earth is the third planet from the", "Sun"),
    ("The Statue of", "Liberty"),
    ("Charles Darwin was an English naturalist who proposed the theory of", "evolution"),
    ("Isaac Newton was an English mathematician and physicist who formulated the laws of", "motion"),
    ("Mount Everest, with an elevation of 8,848 meters, is Earth's highest", "mountain"),
    ("The Pacific Ocean is the largest and deepest of Earth's five oceanic", "divisions"),
    ("William Shakespeare was an English playwright who wrote plays such as Hamlet, Macbeth, and Romeo and", "Juliet"),
]


def main():
    ckpt = "checkpoints/1.1_step_150000.pt"
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
        is_match = expected.lower() in new.lower()[:50]
        marker = "✓" if is_match else "✗"
        if is_match:
            correct += 1
        print(f"[{i:2d}] {marker} '...{p[-50:] if len(p) > 50 else p}' -> '{new[:35]}' (want: {expected})")

    print(f"\nScore: {correct}/{len(PROMPTS)} = {100*correct/len(PROMPTS):.0f}%")


if __name__ == "__main__":
    main()
