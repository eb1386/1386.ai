#!/usr/bin/env python3
# quick CPU text-completion test of a base (pretrain) checkpoint.
# forces CPU so it never touches the GPU that training is using.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import glob
import re
import sys
from pathlib import Path

import torch
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint
from src.inference.generate import generate

DEFAULT_PROMPTS = [
    "The capital of France is",
    "The Earth orbits around the",
    "Water is made up of hydrogen and",
    "The largest planet in our solar system is",
    "Once upon a time, there was a",
    "To boil an egg, first you",
    "In Python, you define a function using the keyword",
    "The opposite of hot is",
]


def latest_ckpt(prefix="1.1_v2_step"):
    files = glob.glob(str(ROOT / "checkpoints" / f"{prefix}_*.pt"))
    if not files:
        return None
    return max(files, key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default="configs/pretrain_1.1_v2.yaml")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--prompt", action="append", default=[])
    args = ap.parse_args()

    ckpt = args.checkpoint or latest_ckpt()
    if not ckpt:
        raise SystemExit("no checkpoint found")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    cfg = load_config(args.config)
    model = Transformer(ModelConfig.from_dict(cfg["model"]))
    step, _ = load_checkpoint(ckpt, model)
    model.eval()
    tok = spm.SentencePieceProcessor()
    tok.load(cfg["data"]["tokenizer_path"])

    prompts = args.prompt or DEFAULT_PROMPTS
    print(f"base checkpoint: {Path(ckpt).name} (step {step}) — greedy CPU completion\n")
    for p in prompts:
        cont = generate(model, tok, p, max_tokens=args.max_tokens, temperature=0.0,
                        top_k=1, top_p=1.0, repetition_penalty=1.2,
                        device=torch.device("cpu"), return_new_only=True)
        print(f"  {p}|{cont.strip()}")
    print()


if __name__ == "__main__":
    main()
