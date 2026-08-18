# test emergent chat / multi-turn behavior on base model (no sft yet)

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

    print(f"loading {ckpt}...")
    model = Transformer(model_cfg).to(device)
    load_checkpoint(ckpt, model)
    model.eval()
    print(f"  {model.count_parameters():,} params\n")

    tok = spm.SentencePieceProcessor()
    tok.load(cfg["data"]["tokenizer_path"])

    print("=" * 70)
    print("CHAT-FORMAT PROMPTS (no sft yet, just probing emergent behavior)")
    print("=" * 70)
    chat_prompts = [
        "User: What is your favorite color?\nAssistant:",
        "User: Hi!\nAssistant: Hello!\nUser: How are you today?\nAssistant:",
        "User: Tell me a fun fact about cats.\nAssistant:",
        "User: I love pizza.\nAssistant: That's great!\nUser: What's your favorite topping?\nAssistant:",
    ]
    for p in chat_prompts:
        print(f"\n--- INPUT ---\n{p}")
        out = generate(
            model, tok, p,
            max_tokens=60,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            device=device,
        )
        # extract only the new continuation
        new = out[len(p):].split("User:")[0].split("\nAssistant:")[0].strip()
        print(f"--- CONTINUATION ---\n{new}")

    print("\n" + "=" * 70)
    print("REASONING PROMPTS (completion-style, easy logic)")
    print("=" * 70)
    reasoning = [
        "If it is raining outside, then the ground is",
        "All dogs are mammals. Rex is a dog. Therefore Rex is a",
        "Sarah is taller than Tom. Tom is taller than Mike. Who is the tallest?",
        "Q: What color is the sky?\nA:",
    ]
    for p in reasoning:
        print(f"\n--- INPUT ---\n{p}")
        out = generate(
            model, tok, p,
            max_tokens=40,
            temperature=0.5,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.2,
            device=device,
        )
        new = out[len(p):].strip()
        print(f"--- CONTINUATION ---\n{new}")


if __name__ == "__main__":
    main()
