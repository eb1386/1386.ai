# test mid-pretrain checkpoint on cpu
# pretrain is base completion only - no chat yet

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint
from src.inference.generate import generate


def main():
    ckpt = "checkpoints/1.1_step_120000.pt"
    cfg = load_config("configs/pretrain_1.1.yaml")
    model_cfg = ModelConfig.from_dict(cfg["model"])
    device = torch.device("cpu")

    print(f"loading {ckpt} on cpu (training holds gpu)...")
    t0 = time.time()
    model = Transformer(model_cfg).to(device)
    load_checkpoint(ckpt, model)
    model.eval()
    print(f"  {model.count_parameters():,} params, loaded in {time.time()-t0:.1f}s")

    tok = spm.SentencePieceProcessor()
    tok.load(cfg["data"]["tokenizer_path"])

    prompts = [
        "The capital of France is",
        "George Washington was the first",
        "Once upon a time, there was a farmer named Bob who",
        "Python is a programming language that",
        "The sun is a star that",
        "Two plus two equals",
        "The largest planet in our solar system is",
    ]

    for p in prompts:
        print(f"\n>>> {p}")
        t0 = time.time()
        out = generate(
            model, tok, p,
            max_tokens=40,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            device=device,
        )
        dt = time.time() - t0
        print(f"    {out}")
        print(f"    [{dt:.1f}s]")


if __name__ == "__main__":
    main()
