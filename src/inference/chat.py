# cli chat interface

import argparse
import sys

import torch
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint
from src.inference.generate import generate

MAX_HISTORY_TOKENS = 768


def format_chat(history: list[dict], tokenizer=None, multiturn=False) -> str:
    # format history into prompt
    if not multiturn:
        last_user = ""
        for msg in reversed(history):
            if msg["role"] == "user":
                last_user = msg["content"]
                break
        return f"User: {last_user}\nAssistant:"

    turns = []
    for msg in history:
        if msg["role"] == "user":
            turns.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            turns.append(f"Assistant: {msg['content']}")
    turns.append("Assistant:")

    prompt = "\n".join(turns)

    if tokenizer:
        tokens = tokenizer.encode(prompt, out_type=int)
        while len(turns) > 3 and len(tokens) > MAX_HISTORY_TOKENS:
            turns.pop(0)
            prompt = "\n".join(turns)
            tokens = tokenizer.encode(prompt, out_type=int)

    return prompt


def main():
    parser = argparse.ArgumentParser(description="1386.ai Chat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--multiturn", action="store_true",
                        help="enable multi-turn context")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    data_cfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg = cfg.get("training", {})
    multiturn = args.multiturn or "1.1" in train_cfg.get("checkpoint_prefix", "")

    print("Loading model...")
    model = Transformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(data_cfg["tokenizer_path"])

    mode_str = "multi-turn" if multiturn else "single-turn"
    print(f"\n1386.ai Chat ({model.count_parameters():,} params, {mode_str})")
    print(f"Device: {device}")
    print("Type 'quit' or 'exit' to leave. Type 'clear' to reset history.\n")

    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[History cleared]\n")
            continue

        history.append({"role": "user", "content": user_input})
        prompt = format_chat(history, tokenizer=tokenizer, multiturn=multiturn)

        full_output = generate(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=1.5,
            device=device,
        )

        response = full_output[len(prompt):].strip()
        for stop in ["\nUser:", "\nSystem:", "\nHuman:", "\nQuestion:", "\n\n\n"]:
            if stop in response:
                response = response[: response.index(stop)]

        response = response.strip()

        if response.rstrip(".").isdigit() and len(response.rstrip(".")) <= 3:
            response = "I'm not sure about that."

        if len(response) > 100 and response[-1] not in ".!?\"'":
            last_end = max(response.rfind("."), response.rfind("!"), response.rfind("?"))
            if last_end > 50:
                response = response[:last_end + 1]

        if not response:
            response = "(empty response)"

        history.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
