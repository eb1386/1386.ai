# perplexity eval

import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import ShardDataset
from src.train.utils import load_config, load_checkpoint


def evaluate_perplexity(
    model: Transformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()
            n_batches += 1
            if max_batches > 0 and n_batches >= max_batches:
                break

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": ppl, "tokens": total_tokens, "batches": n_batches}


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--max_batches", type=int, default=0, help="0 = all batches")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    data_cfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)

    dataset = ShardDataset(shard_dir=data_cfg["shard_dir"], split="val", seq_len=data_cfg["seq_len"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Evaluating perplexity on {len(dataset)} sequences...")
    results = evaluate_perplexity(model, loader, device, args.max_batches)
    print(f"  Loss:       {results['loss']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Tokens:     {results['tokens']:,}")


if __name__ == "__main__":
    main()
