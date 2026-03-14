# training loop

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import ShardDataset, StreamingShardDataset
from src.train.scheduler import CosineScheduler
from src.train.utils import load_config, save_checkpoint, load_checkpoint, JSONLLogger


def main():
    parser = argparse.ArgumentParser(description="train 1386.ai")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--finetune", default=None, help="Path to checkpoint to fine-tune (loads weights only, resets step/optimizer)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = Transformer(model_cfg).to(device)
    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing = True
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=train_cfg["weight_decay"],
        fused=device.type == "cuda",
    )

    scheduler = CosineScheduler(
        learning_rate=train_cfg["learning_rate"],
        min_lr=train_cfg["min_lr"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
    )

    start_step = 0
    if args.finetune:
        print(f"Fine-tuning from {args.finetune}...")
        load_checkpoint(args.finetune, model)  # Load weights only, no optimizer
        print(f"Loaded pretrained weights (step reset to 0)")
    elif args.resume:
        print(f"Resuming from {args.resume}...")
        start_step, _ = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed at step {start_step}")

    use_loss_mask = train_cfg.get("use_loss_mask", False)
    if use_loss_mask:
        print("Loss masking ENABLED — only training on assistant responses")

    ckpt_prefix = train_cfg.get("checkpoint_prefix", "step")

    use_streaming = True
    try:
        dataset = StreamingShardDataset(
            shard_dir=data_cfg["shard_dir"],
            split="train",
            seq_len=data_cfg["seq_len"],
            use_loss_mask=use_loss_mask,
        )
    except FileNotFoundError:
        print("WARNING: No shards found. Run scripts/build_shards.py first.")
        return

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["micro_batch_size"],
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    try:
        val_dataset = ShardDataset(
            shard_dir=data_cfg["shard_dir"],
            split="val",
            seq_len=data_cfg["seq_len"],
            use_loss_mask=use_loss_mask,
        )
        val_loader = DataLoader(val_dataset, batch_size=train_cfg["micro_batch_size"], shuffle=False)
    except FileNotFoundError:
        val_loader = None

    use_amp = train_cfg.get("precision", "bf16") == "bf16" and device.type == "cuda"
    dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = None  # bf16 doesn't need GradScaler

    logger = JSONLLogger("logs/train.jsonl")
    grad_accum = train_cfg["gradient_accumulation"]
    max_steps = train_cfg["max_steps"]

    print(f"\nTraining config:")
    print(f"  Max steps:       {max_steps}")
    print(f"  Micro batch:     {train_cfg['micro_batch_size']}")
    print(f"  Grad accum:      {grad_accum}")
    print(f"  Effective batch: {train_cfg['micro_batch_size'] * grad_accum}")
    print(f"  Seq len:         {data_cfg['seq_len']}")
    print(f"  Precision:       {'bf16' if use_amp else 'fp32'}")
    print(f"  Grad checkpoint: {train_cfg.get('gradient_checkpointing', False)}")
    print(f"  Loss masking:    {use_loss_mask}")
    print(f"  Ckpt prefix:     {ckpt_prefix}")
    print()

    model.train()
    data_iter = iter(loader)
    step = start_step
    accum_loss = 0.0
    t0 = time.time()
    tokens_processed = 0

    optimizer.zero_grad()

    while step < max_steps:
        for micro_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            if use_loss_mask and len(batch) == 3:
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
            else:
                x, y = batch[0], batch[1]
                x, y = x.to(device), y.to(device)
                mask = None

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                logits = model(x)
                if mask is not None:
                    loss_per_token = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
                    )
                    mask_flat = mask.view(-1)
                    masked_loss = (loss_per_token * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                    loss = masked_loss / grad_accum
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum

            loss.backward()
            accum_loss += loss.item()
            tokens_processed += x.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg["max_grad_norm"]
        )

        lr = scheduler.get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if step % train_cfg["log_interval"] == 0:
            dt = time.time() - t0
            tok_per_sec = tokens_processed / dt if dt > 0 else 0
            avg_loss = accum_loss / train_cfg["log_interval"]
            log_data = {
                "step": step,
                "loss": round(avg_loss, 4),
                "lr": round(lr, 8),
                "grad_norm": round(grad_norm.item(), 4) if isinstance(grad_norm, torch.Tensor) else round(grad_norm, 4),
                "tok_per_sec": round(tok_per_sec),
                "dt": round(dt, 2),
            }
            logger.log(log_data)
            print(
                f"step {step:>6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                f"grad_norm {log_data['grad_norm']:.2f} | "
                f"{tok_per_sec:,.0f} tok/s"
            )
            accum_loss = 0.0
            tokens_processed = 0
            t0 = time.time()

        if val_loader is not None and step % train_cfg["eval_interval"] == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    if use_loss_mask and len(val_batch) == 3:
                        vx, vy, vmask = val_batch
                        vx, vy, vmask = vx.to(device), vy.to(device), vmask.to(device)
                    else:
                        vx, vy = val_batch[0], val_batch[1]
                        vx, vy = vx.to(device), vy.to(device)
                        vmask = None

                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                        vlogits = model(vx)
                        if vmask is not None:
                            vloss_per_tok = F.cross_entropy(
                                vlogits.view(-1, vlogits.size(-1)), vy.view(-1), reduction="none"
                            )
                            vmask_flat = vmask.view(-1)
                            vloss = (vloss_per_tok * vmask_flat).sum() / (vmask_flat.sum() + 1e-8)
                        else:
                            vloss = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                    val_loss += vloss.item()
                    val_steps += 1
                    if val_steps >= 50:
                        break
            avg_val = val_loss / val_steps
            print(f"  [eval] step {step} | val_loss {avg_val:.4f} | val_ppl {2.71828**avg_val:.2f}")
            logger.log({"step": step, "val_loss": round(avg_val, 4), "val_ppl": round(2.71828**avg_val, 2)})
            model.train()

        if step % train_cfg["checkpoint_interval"] == 0:
            ckpt_path = f"checkpoints/{ckpt_prefix}_{step}.pt"
            save_checkpoint(model, optimizer, step, cfg, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    save_checkpoint(model, optimizer, step, cfg, "checkpoints/final.pt")
    print(f"\nTraining complete. Final checkpoint: checkpoints/final.pt")
    logger.close()


if __name__ == "__main__":
    main()
