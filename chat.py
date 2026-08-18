# terminal chat for plasma 1.1

import torch
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint
from src.inference.generate import generate

def main():
    cfg = load_config("configs/finetune_1.1.yaml")
    model_cfg = ModelConfig.from_dict(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  loading plasma 1.1 on {device}...")
    model = Transformer(model_cfg).to(device)
    load_checkpoint("checkpoints/finetune_1.1_final.pt", model)
    model.eval()
    print(f"  {model.count_parameters():,} parameters\n")

    tokenizer = spm.SentencePieceProcessor()
    tok_path = cfg["data"]["tokenizer_path"]
    tokenizer.load(tok_path)

    history = []
    print("  plasma 1.1 — type 'quit' to exit, 'clear' to reset\n")

    while True:
        try:
            user_input = input("you: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history.clear()
            print("  (history cleared)\n")
            continue

        history.append(f"User: {user_input}")

        # build prompt
        turns = history.copy()
        turns.append("Assistant: ")
        prompt = "\n".join(turns)

        # trim history
        tokens = tokenizer.encode(prompt, out_type=int)
        while len(turns) > 3 and len(tokens) > 768:
            turns.pop(0)
            prompt = "\n".join(turns)
            tokens = tokenizer.encode(prompt, out_type=int)

        output = generate(
            model, tokenizer, prompt,
            max_tokens=200,
            temperature=0.3,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.3,
            device=device,
            return_new_only=True,
        )

        # extract response
        response = output.strip()
        for stop in ["\nUser:", "\nAssistant:", "\nSystem:", "\nHuman:", "\n\n\n"]:
            if stop in response:
                response = response[:response.index(stop)]
        response = response.strip()

        if not response:
            response = "(empty response)"

        history.append(f"Assistant: {response}")
        print(f"1386: {response}\n")

if __name__ == "__main__":
    main()
