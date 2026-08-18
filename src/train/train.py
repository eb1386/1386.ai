# training loop

import argparse
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sentencepiece as spm
import yaml

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import ShardDataset, StreamingShardDataset
from src.train.scheduler import build_scheduler
from src.train.utils import load_config, save_checkpoint, load_checkpoint, JSONLLogger


def rotate_checkpoints(ckpt_prefix: str, keep_last: int):
    """delete old step checkpoints, keep newest n."""
    if keep_last <= 0:
        return
    ckpt_dir = Path("checkpoints")
    pat = re.compile(rf"^{re.escape(ckpt_prefix)}_(\d+)\.pt$")
    found = []
    for p in ckpt_dir.glob(f"{ckpt_prefix}_*.pt"):
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    for _, p in found[:-keep_last]:
        try:
            p.unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="train 1386.ai")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--finetune", default=None, help="Path to checkpoint to fine-tune (loads weights only, resets step/optimizer)")
    parser.add_argument("--log-path", default=None, help="Override JSONL training log path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    shard_meta_path = Path(data_cfg["shard_dir"]) / "meta.yaml"
    if shard_meta_path.exists():
        with open(shard_meta_path, "r", encoding="utf-8") as f:
            shard_meta = yaml.safe_load(f) or {}
        if shard_meta.get("vocab_size") and int(shard_meta["vocab_size"]) != int(model_cfg.vocab_size):
            raise ValueError(
                f"Shard vocab_size {shard_meta['vocab_size']} != config vocab_size {model_cfg.vocab_size}"
            )
        if shard_meta.get("tokenizer") and Path(data_cfg["tokenizer_path"]).name != shard_meta["tokenizer"]:
            raise ValueError(
                f"Shard tokenizer {shard_meta['tokenizer']} != config tokenizer {Path(data_cfg['tokenizer_path']).name}"
            )

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(data_cfg["tokenizer_path"])
    if tokenizer.get_piece_size() != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab {tokenizer.get_piece_size()} != model vocab {model_cfg.vocab_size}"
        )

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

    # depth-scaled residual init: scale output projections by 1/sqrt(2*n_layers)
    # so residual-stream variance does not grow with depth (GPT-2/Llama practice)
    if train_cfg.get("depth_scaled_init", False):
        import math as _math
        scale = 1.0 / _math.sqrt(2 * model_cfg.num_layers)
        n_scaled = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith("attn.o_proj.weight") or name.endswith("ffn.down_proj.weight"):
                    p.mul_(scale)
                    n_scaled += 1
        print(f"Depth-scaled init applied to {n_scaled} residual projections (x{scale:.4f})")

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("tok_emb.weight") or name.endswith("output.weight"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    betas = tuple(train_cfg.get("betas", (0.9, 0.95)))
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if opt_name in ("adamw8bit", "adamw_8bit"):
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            param_groups, lr=train_cfg["learning_rate"], betas=betas,
        )
        print("Optimizer: bitsandbytes AdamW8bit")
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=train_cfg["learning_rate"],
            betas=betas,
            fused=device.type == "cuda",
        )

    scheduler = build_scheduler(train_cfg)
    print(f"LR schedule: {type(scheduler).__name__} "
          f"(peak {train_cfg['learning_rate']:.2e} -> min {train_cfg['min_lr']:.2e})")

    start_step = 0
    if args.finetune:
        print(f"Fine-tuning from {args.finetune}...")
        load_checkpoint(args.finetune, model)  # Load weights only, no optimizer
        print(f"Loaded pretrained weights (step reset to 0)")
    elif args.resume:
        print(f"Resuming from {args.resume}...")
        start_step, _ = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed at step {start_step}")

    # compile only the forward path; `model` stays the source of truth for
    # checkpointing, clipping and the optimizer so state dict keys stay clean
    fwd = model
    if train_cfg.get("compile", False):
        try:
            fwd = torch.compile(model)
            print("torch.compile ENABLED (first steps will be slow while it warms up)")
        except Exception as e:
            print(f"torch.compile unavailable ({type(e).__name__}: {e}); continuing uncompiled")

    use_loss_mask = train_cfg.get("use_loss_mask", False)
    if use_loss_mask:
        print("Loss masking ENABLED — only training on assistant responses")

    # v4+ shards mark conversation starts with bos; a block-diagonal mask
    # stops attention crossing between packed conversations
    doc_mask_bos = train_cfg.get("doc_attention_bos_id", None)
    if doc_mask_bos is not None:
        print(f"Doc attention masking ENABLED (bos id {doc_mask_bos})")

    # small logit regularizer; keeps logsumexp near 0, standard for stability
    z_loss_coef = float(train_cfg.get("z_loss", 0.0))
    if z_loss_coef:
        print(f"z-loss ENABLED (coef {z_loss_coef})")

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

    # validation on a FIXED RANDOM subset. reading the first N sequences in file
    # order samples the head of the corpus (one domain) and 100 sequences is far
    # too noisy to compare checkpoints -- take a seeded random subset instead.
    val_sequences = int(train_cfg.get("val_sequences", 2048))
    val_batch_size = int(train_cfg.get("val_batch_size", 8))
    try:
        val_dataset = ShardDataset(
            shard_dir=data_cfg["shard_dir"],
            split="val",
            seq_len=data_cfg["seq_len"],
            use_loss_mask=use_loss_mask,
        )
        n_val = len(val_dataset)
        if val_sequences < n_val:
            import numpy as _np
            idx = _np.random.default_rng(1386).permutation(n_val)[:val_sequences]
            val_dataset = torch.utils.data.Subset(val_dataset, idx.tolist())
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        print(f"Validation: {min(val_sequences, n_val):,} sequences (of {n_val:,}), batch {val_batch_size}")
    except FileNotFoundError:
        val_loader = None

    use_amp = train_cfg.get("precision", "bf16") == "bf16" and device.type == "cuda"
    dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = None  # bf16 doesn't need GradScaler

    logger = JSONLLogger(args.log_path or train_cfg.get("log_path", "logs/train.jsonl"))
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
            elif use_loss_mask:
                raise RuntimeError("use_loss_mask=True but dataset returned no loss mask")
            else:
                x, y = batch[0], batch[1]
                x, y = x.to(device), y.to(device)
                mask = None

            attn_mask = None
            if doc_mask_bos is not None:
                attn_mask = model.doc_attention_mask(x, doc_mask_bos)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                logits = fwd(x) if attn_mask is None else fwd(x, attn_mask=attn_mask)
                if mask is not None:
                    loss_per_token = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
                    )
                    mask_flat = mask.view(-1)
                    masked_loss = (loss_per_token * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                    loss = masked_loss
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                if z_loss_coef:
                    z = torch.logsumexp(logits.float(), dim=-1)
                    if mask is not None:
                        zl = (z.pow(2).view(-1) * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                    else:
                        zl = z.pow(2).mean()
                    loss = loss + z_loss_coef * zl
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
                    elif use_loss_mask:
                        raise RuntimeError("use_loss_mask=True but val dataset returned no loss mask")
                    else:
                        vx, vy = val_batch[0], val_batch[1]
                        vx, vy = vx.to(device), vy.to(device)
                        vmask = None

                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                        vlogits = fwd(vx)
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
            avg_val = val_loss / val_steps
            print(f"  [eval] step {step} | val_loss {avg_val:.4f} | val_ppl {2.71828**avg_val:.2f}")
            logger.log({"step": step, "val_loss": round(avg_val, 4), "val_ppl": round(2.71828**avg_val, 2)})
            model.train()

        if step % train_cfg["checkpoint_interval"] == 0:
            ckpt_path = f"checkpoints/{ckpt_prefix}_{step}.pt"
            save_checkpoint(model, optimizer, step, cfg, ckpt_path)
            rotate_checkpoints(ckpt_prefix, train_cfg.get("keep_last_checkpoints", 4))
            print(f"  Saved checkpoint: {ckpt_path}")

    # derive final name from prefix; treat any prefix containing "_ft" as a finetune
    is_finetune = bool(args.finetune) or "_ft" in ckpt_prefix
    prefix = ckpt_prefix.rstrip("_").replace("_step", "").replace("_ft", "")
    final_path = f"checkpoints/finetune_{prefix}_final.pt" if is_finetune else f"checkpoints/pretrain_{prefix}_final.pt"
    save_checkpoint(model, optimizer, step, cfg, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    logger.close()


if __name__ == "__main__":
    main()
