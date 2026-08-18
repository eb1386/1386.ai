# cli chat interface

import argparse
import sys

import torch
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint
from src.inference.generate import generate
from src.inference.template import build_prompt_ids, stop_sequences, penalty_exclude


def main():
    parser = argparse.ArgumentParser(description="1386.ai Chat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/tiny.yaml")
    # match the web serving defaults so cli observations are representative
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
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

        # token-level template, exactly as the sft data was encoded
        prior = history if multiturn else []
        ids = build_prompt_ids(
            tokenizer, user_input, prior,
            max_prompt_tokens=model_cfg.max_seq_len - args.max_tokens)
        history.append({"role": "user", "content": user_input})

        response, meta = generate(
            model, tokenizer, "",
            prompt_ids=ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            penalty_window=128,
            penalty_exclude=penalty_exclude(tokenizer),
            stop_id_seqs=stop_sequences(tokenizer),
            device=device,
            return_new_only=True,
            return_meta=True,
        )

        response = response.strip()
        for stop in ["\nUser:", "\n User:", "\nAssistant:", "\n Assistant:",
                     "\nSystem:", "\nHuman:", "\nQuestion:", "\n\n\n"]:
            if stop in response:
                response = response[: response.index(stop)]

        response = response.strip()

        # only trim dangling fragments from a hard length cutoff
        if meta["stop"] in ("length", "context") and len(response) > 100 \
                and response[-1] not in ".!?\"'":
            last_end = max(response.rfind("."), response.rfind("!"), response.rfind("?"))
            if last_end > 50:
                response = response[:last_end + 1]

        if not response:
            response = "(empty response)"

        history.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
