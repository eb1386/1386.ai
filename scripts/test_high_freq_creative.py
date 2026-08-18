# high-frequency facts + creative writing test

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


HIGH_FREQ = [
    "The first president of the United States was",
    "The largest country in the world by area is",
    "The author of Romeo and Juliet was",
    "The fastest land animal is the",
    "Albert Einstein is famous for his theory of",
    "The currency of Japan is the",
    "The capital of Japan is",
    "The longest river in the world is the",
    "The tallest mountain in the world is",
    "The Great Wall is located in the country of",
    "The Mona Lisa was painted by",
    "The boiling point of water is",
]

CREATIVE = [
    ("Short story (60 tokens)",
     "The dragon hadn't seen a human in a hundred years, but tonight"),
    ("Story continuation (80 tokens)",
     "Once upon a time, in a small village by the sea, there lived a young girl named Mira who"),
    ("Essay opener (80 tokens)",
     "Exercise is important for your health because"),
    ("Descriptive (60 tokens)",
     "The forest at dawn was quiet. Mist rose from the ground, and"),
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

    print("=" * 70)
    print("HIGH-FREQUENCY FACTS")
    print("=" * 70)
    for i, p in enumerate(HIGH_FREQ, 1):
        t0 = time.time()
        out = generate(
            model, tok, p,
            max_tokens=20,
            temperature=0.3,
            top_k=10,
            top_p=0.85,
            repetition_penalty=1.2,
            device=device,
        )
        new = out[len(p):].strip().split("\n")[0]
        print(f"\n[{i:2d}] {p}\n     -> {new}  [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 70)
    print("CREATIVE WRITING")
    print("=" * 70)
    for label, p in CREATIVE:
        max_tok = 60 if "60" in label else 80
        t0 = time.time()
        out = generate(
            model, tok, p,
            max_tokens=max_tok,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.3,
            device=device,
        )
        print(f"\n--- {label} ---\n{out}\n  [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
