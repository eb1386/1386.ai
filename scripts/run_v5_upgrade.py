#!/usr/bin/env python3
# 1.0 training pipeline

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ─── Project root ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── Paths ───────────────────────────────────────────────────────────────
V4_RAW         = ROOT / "data" / "raw_v4"
V5_PRETRAIN    = ROOT / "data" / "shards_1.0"
V5_INSTRUCT    = ROOT / "data" / "instruct_shards_1.0"
TOKENIZER_PATH = ROOT / "data" / "tokenizer_v4.model"
CKPT_DIR       = ROOT / "checkpoints"
LOG_DIR        = ROOT / "logs"

PRETRAIN_CFG   = ROOT / "configs" / "pretrain_1.0.yaml"
FINETUNE_CFG   = ROOT / "configs" / "finetune_1.0.yaml"

VOCAB_SIZE = 32000
SEQ_LEN = 1024

# Target: ~12-15GB of text total (existing ~3GB + ~10GB new)
FINEWEB_TARGET_CHARS = 10_000_000_000  # ~10GB of new text → ~3B tokens


# ═════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════
def banner(msg):
    print(f"\n{'=' * 64}")
    print(f"  {msg}")
    print(f"{'=' * 64}\n")


def elapsed_str(seconds):
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ═════════════════════════════════════════════════════════════════════════
# Stage 0: Clean up old files
# ═════════════════════════════════════════════════════════════════════════
def stage_cleanup():
    banner("1.0 Stage 0: Clean up old files (~65 GB)")

    freed = 0

    # Delete old v4 step checkpoints (keep pretrain_v4_final.pt)
    old_step_ckpts = list(CKPT_DIR.glob("step_*.pt"))
    if old_step_ckpts:
        for ckpt in old_step_ckpts:
            size = ckpt.stat().st_size
            ckpt.unlink()
            freed += size
        print(f"  Deleted {len(old_step_ckpts)} old V4 step checkpoints")

    # Delete old v1-v3 checkpoints
    for name in ["pretrain_final.pt", "pretrain_v2.pt"]:
        p = CKPT_DIR / name
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
            print(f"  Deleted old checkpoint: {name}")

    # Delete finetune_v4_final.pt (duplicate of final.pt)
    ft_v4 = CKPT_DIR / "finetune_v4_final.pt"
    if ft_v4.exists():
        freed += ft_v4.stat().st_size
        ft_v4.unlink()
        print(f"  Deleted duplicate: finetune_v4_final.pt")

    # Delete old final.pt
    old_final = CKPT_DIR / "final.pt"
    if old_final.exists():
        freed += old_final.stat().st_size
        old_final.unlink()
        print(f"  Deleted old final.pt")

    # Delete old v1-v3 data directories
    old_dirs = [
        ROOT / "data" / "shards",
        ROOT / "data" / "instruct_shards",
        ROOT / "data" / "raw",
    ]
    for d in old_dirs:
        if d.exists():
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(str(d))
            freed += size
            print(f"  Deleted old directory: {d.name}/")

    # Delete old v1-v3 tokenizer files
    for name in ["tokenizer.model", "tokenizer.vocab"]:
        p = ROOT / "data" / name
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
            print(f"  Deleted old tokenizer: {name}")

    # Delete old shard directories
    for d in [ROOT / "data" / "shards_v4", ROOT / "data" / "instruct_shards_v4"]:
        if d.exists():
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(str(d))
            freed += size
            print(f"  Deleted old shards: {d.name}/")

    if freed > 0:
        print(f"\n  Freed {freed / 1e9:.1f} GB of disk space")
    else:
        print("  Nothing to clean up")


# ═════════════════════════════════════════════════════════════════════════
# Stage 1: Download FineWeb-Edu data
# ═════════════════════════════════════════════════════════════════════════
def stage_download():
    banner("1.0 Stage 1: Download FineWeb-Edu data")

    V4_RAW.mkdir(parents=True, exist_ok=True)
    fineweb_path = V4_RAW / "fineweb_edu_corpus.txt"

    # Check if already downloaded
    if fineweb_path.exists() and fineweb_path.stat().st_size >= FINEWEB_TARGET_CHARS * 0.8:
        print(f"[skip] FineWeb-Edu corpus exists ({fineweb_path.stat().st_size / 1e9:.2f} GB)")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)

    print(f"Downloading FineWeb-Edu (target: {FINEWEB_TARGET_CHARS / 1e9:.0f} GB of text)...")
    print("  This is high-quality educational web text from HuggingFace.")
    print("  Streaming — may take 1-2 hours depending on internet speed.")
    print()

    total_chars = 0
    doc_count = 0
    t0 = time.time()

    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        with open(fineweb_path, "w", encoding="utf-8") as f:
            for example in ds:
                text = example["text"].strip()
                if len(text) < 100:
                    continue
                f.write(text)
                f.write("\n\n")
                total_chars += len(text) + 2
                doc_count += 1

                if doc_count % 50000 == 0:
                    speed = total_chars / (time.time() - t0 + 0.1) / 1e6
                    print(f"  {doc_count:,} docs | {total_chars / 1e9:.2f} GB | "
                          f"{speed:.1f} MB/s")

                if total_chars >= FINEWEB_TARGET_CHARS:
                    break

        dt = time.time() - t0
        print(f"\nFineWeb-Edu download complete:")
        print(f"  Documents: {doc_count:,}")
        print(f"  Size: {fineweb_path.stat().st_size / 1e9:.2f} GB")
        print(f"  Time: {elapsed_str(dt)}")

    except Exception as e:
        print(f"\nERROR downloading FineWeb-Edu: {e}")
        print("Trying alternative: FineWeb (non-edu)...")
        try:
            ds = load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

            with open(fineweb_path, "w", encoding="utf-8") as f:
                for example in ds:
                    text = example["text"].strip()
                    if len(text) < 100:
                        continue
                    f.write(text)
                    f.write("\n\n")
                    total_chars += len(text) + 2
                    doc_count += 1

                    if doc_count % 50000 == 0:
                        speed = total_chars / (time.time() - t0 + 0.1) / 1e6
                        print(f"  {doc_count:,} docs | {total_chars / 1e9:.2f} GB | "
                              f"{speed:.1f} MB/s")

                    if total_chars >= FINEWEB_TARGET_CHARS:
                        break

            print(f"\nFineWeb download complete: {doc_count:,} docs, "
                  f"{fineweb_path.stat().st_size / 1e9:.2f} GB")

        except Exception as e2:
            print(f"ERROR: Could not download FineWeb either: {e2}")
            print("Please check your internet connection and try again.")
            sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════
# Stage 2: Build pretrain shards (combined old + new data)
# ═════════════════════════════════════════════════════════════════════════
def stage_build_pretrain_shards():
    banner("1.0 Stage 2: Build pretrain shards (combined corpus)")

    meta_path = V5_PRETRAIN / "meta.yaml"
    if meta_path.exists():
        print(f"[skip] 1.0 pretrain shards exist: {V5_PRETRAIN}")
        return

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    eos_id = sp.eos_id()

    # Collect all pretrain text files
    corpus_files = []
    owt_path = V4_RAW / "pretrain_corpus.txt"
    fineweb_path = V4_RAW / "fineweb_edu_corpus.txt"

    if owt_path.exists():
        corpus_files.append(("OpenWebText+Wiki", owt_path))
    if fineweb_path.exists():
        corpus_files.append(("FineWeb-Edu", fineweb_path))

    if not corpus_files:
        print("ERROR: No pretrain corpus files found. Run stage 1 first.")
        sys.exit(1)

    total_size = sum(f.stat().st_size for _, f in corpus_files)
    print(f"Corpus files ({total_size / 1e9:.1f} GB total):")
    for name, path in corpus_files:
        print(f"  {name}: {path.stat().st_size / 1e9:.2f} GB")

    pack_len = SEQ_LEN + 1
    V5_PRETRAIN.mkdir(parents=True, exist_ok=True)

    # Tokenize all files
    print("\nTokenizing combined corpus...")
    t0 = time.time()
    token_chunks = []
    total_tokens = 0
    CHUNK_LINES = 2000

    for name, path in corpus_files:
        print(f"\n  Processing {name} ({path.stat().st_size / 1e9:.2f} GB)...")
        file_tokens = 0
        line_buf = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_buf.append(line.rstrip("\n"))
                if len(line_buf) >= CHUNK_LINES:
                    text = "\n".join(line_buf)
                    tokens = sp.encode(text, out_type=int)
                    tokens.append(eos_id)
                    token_chunks.append(np.array(tokens, dtype=np.uint16))
                    total_tokens += len(tokens)
                    file_tokens += len(tokens)
                    line_buf = []
                    if len(token_chunks) % 500 == 0:
                        print(f"    {total_tokens:,} tokens "
                              f"({total_tokens / 1e9:.3f}B)...")

            if line_buf:
                text = "\n".join(line_buf)
                tokens = sp.encode(text, out_type=int)
                tokens.append(eos_id)
                token_chunks.append(np.array(tokens, dtype=np.uint16))
                total_tokens += len(tokens)
                file_tokens += len(tokens)

        print(f"    {name}: {file_tokens:,} tokens ({file_tokens / 1e9:.3f}B)")

    dt = time.time() - t0
    print(f"\n  Total: {total_tokens:,} tokens ({total_tokens / 1e9:.3f}B) "
          f"in {elapsed_str(dt)}")

    # Concatenate and pack
    print("  Packing into sequences...")
    all_tokens = np.concatenate(token_chunks)
    del token_chunks
    gc.collect()

    n_sequences = len(all_tokens) // pack_len
    trimmed = n_sequences * pack_len
    packed = all_tokens[:trimmed].reshape(n_sequences, pack_len)
    del all_tokens
    gc.collect()

    # Shuffle
    print(f"  Shuffling {n_sequences:,} sequences...")
    rng = np.random.default_rng(42)
    rng.shuffle(packed)

    # Split train/val (95/5)
    n_val = max(1, n_sequences // 20)
    n_train = n_sequences - n_val

    # Write train shards (100K sequences each)
    seqs_per_shard = 100_000
    shard_idx = 0
    for start in range(0, n_train, seqs_per_shard):
        end = min(start + seqs_per_shard, n_train)
        path = V5_PRETRAIN / f"train_{shard_idx:04d}.bin"
        packed[start:end].tofile(str(path))
        print(f"  {path.name}: {end - start:,} seqs "
              f"({(end - start) * pack_len / 1e6:.1f}M tokens)")
        shard_idx += 1

    # Write val shard
    val_path = V5_PRETRAIN / "val_0000.bin"
    packed[n_train:].tofile(str(val_path))
    print(f"  val_0000.bin: {n_val:,} seqs")

    # Metadata
    meta = {
        "total_tokens": int(total_tokens),
        "seq_len": SEQ_LEN,
        "pack_len": pack_len,
        "n_train_sequences": int(n_train),
        "n_val_sequences": int(n_val),
        "n_train_shards": shard_idx,
        "dtype": "uint16",
    }
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)

    del packed
    gc.collect()
    print(f"\nV5 pretrain shards: {n_train:,} train + {n_val:,} val sequences")
    print(f"  Total tokens: {total_tokens / 1e9:.3f}B")
    print(f"  Train shards: {shard_idx}")


# ═════════════════════════════════════════════════════════════════════════
# Stage 3: Continue pretraining
# ═════════════════════════════════════════════════════════════════════════
def stage_pretrain():
    banner("1.0 Stage 3: Continue pretraining (100k more steps)")

    pretrain_ckpt = CKPT_DIR / "pretrain_1.0_final.pt"
    if pretrain_ckpt.exists():
        print(f"[skip] 1.0 pretrain checkpoint exists: {pretrain_ckpt}")
        return

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Find the best starting checkpoint
    resume_from = None

    # Check for step checkpoints first (if resuming)
    v5_ckpts = sorted(CKPT_DIR.glob("1.0_step_*.pt"))
    if v5_ckpts:
        resume_from = v5_ckpts[-1]
        print(f"Resuming 1.0 pretraining from: {resume_from}")
    else:
        # Start from v4 pretrain checkpoint
        v4_pretrain = CKPT_DIR / "pretrain_v4_final.pt"
        if v4_pretrain.exists():
            print(f"Continuing from V4 pretrain checkpoint: {v4_pretrain}")
            print("  (loading weights only, resetting optimizer + step counter)")
            resume_from = v4_pretrain
        else:
            # Fall back to any step checkpoint
            step_ckpts = sorted(CKPT_DIR.glob("step_*.pt"))
            if step_ckpts:
                resume_from = step_ckpts[-1]
                print(f"Using fallback checkpoint: {resume_from}")
            else:
                print("WARNING: No checkpoint found. Starting pretraining from scratch.")

    print(f"  Config: {PRETRAIN_CFG}")
    print(f"  Target: 100,000 steps (~30 hours on RTX 5080)")
    print(f"  Checkpoints saved every 5000 steps to {CKPT_DIR}/")
    print(f"  You can stop and resume anytime.")
    print()

    # Build command
    cmd = [sys.executable, "-m", "src.train.train",
           "--config", str(PRETRAIN_CFG)]

    # Use --finetune for v4 checkpoint (load weights only, reset optimizer)
    # Use --resume for step checkpoints (load everything, continue)
    if resume_from and "1.0_step_" in resume_from.name:
        cmd.extend(["--resume", str(resume_from)])
    elif resume_from:
        cmd.extend(["--finetune", str(resume_from)])

    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print(f"WARNING: Training exited with code {result.returncode}")
        print("  Checking for latest checkpoint...")

    # Rename final.pt → pretrain_1.0_final.pt
    final = CKPT_DIR / "final.pt"
    if final.exists():
        shutil.copy2(str(final), str(pretrain_ckpt))
        print(f"\nV5 pretrain checkpoint saved: {pretrain_ckpt}")
    else:
        v5_ckpts = sorted(CKPT_DIR.glob("1.0_step_*.pt"))
        if v5_ckpts:
            latest = v5_ckpts[-1]
            shutil.copy2(str(latest), str(pretrain_ckpt))
            print(f"\nUsing latest 1.0 checkpoint: {latest} → {pretrain_ckpt}")
        else:
            print("ERROR: No checkpoint found after pretraining!")
            sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════
# Stage 4: Build instruct shards WITH loss masking
# ═════════════════════════════════════════════════════════════════════════
def stage_build_instruct_shards():
    banner("1.0 Stage 4: Build instruct shards (with loss masking)")

    meta_path = V5_INSTRUCT / "meta.yaml"
    if meta_path.exists():
        print(f"[skip] 1.0 instruct shards exist: {V5_INSTRUCT}")
        return

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    eos_id = sp.eos_id()

    instruct_path = V4_RAW / "instruct_corpus.jsonl"
    if not instruct_path.exists():
        print("ERROR: No instruct corpus. Run the V4 pipeline stage 2 first.")
        sys.exit(1)

    pack_len = SEQ_LEN + 1
    V5_INSTRUCT.mkdir(parents=True, exist_ok=True)

    # Tokenize conversations with loss masks
    print("Tokenizing instruction data with loss masks...")
    t0 = time.time()

    # We'll collect (tokens, mask) pairs per conversation
    all_tokens = []
    all_masks = []
    total_tokens = 0
    n_convs = 0

    # Tokenize the "Assistant:" prefix to find where responses start
    assistant_prefix_tokens = sp.encode("Assistant:", out_type=int)

    with open(instruct_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data["text"]
            except (json.JSONDecodeError, KeyError):
                continue

            # Tokenize the full conversation
            tokens = sp.encode(text, out_type=int)
            if len(tokens) < 4:
                continue

            # Create mask: 1 for assistant tokens, 0 for user/system tokens
            # Find "Assistant:" boundary in the token sequence
            mask = np.zeros(len(tokens), dtype=np.uint8)

            # Tokenize just the user part to find where assistant starts
            # The format is "User: <question>\nAssistant: <answer>"
            asst_pos = text.find("Assistant:")
            if asst_pos >= 0:
                # Tokenize the user prefix to get the split point
                user_prefix = text[:asst_pos]
                user_tokens = sp.encode(user_prefix, out_type=int)
                # Everything from the assistant position onward gets mask=1
                # Include "Assistant:" itself so the model learns to generate it
                split_idx = len(user_tokens)
                mask[split_idx:] = 1
            else:
                # If no "Assistant:" found, mask everything (shouldn't happen)
                mask[:] = 1

            # Add EOS token (should be predicted = mask 1)
            tokens.append(eos_id)
            mask = np.append(mask, 1)

            all_tokens.extend(tokens)
            all_masks.extend(mask)
            total_tokens += len(tokens)
            n_convs += 1

            if n_convs % 100000 == 0:
                print(f"  {n_convs:,} conversations | "
                      f"{total_tokens:,} tokens ({total_tokens / 1e6:.1f}M)")

    dt = time.time() - t0
    print(f"  Tokenized {n_convs:,} conversations → "
          f"{total_tokens:,} tokens ({total_tokens / 1e6:.1f}M) "
          f"in {elapsed_str(dt)}")

    # Pack into fixed-length sequences
    print("  Packing into sequences...")
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    all_masks = np.array(all_masks, dtype=np.uint8)

    n_sequences = len(all_tokens) // pack_len
    trimmed = n_sequences * pack_len

    packed_tokens = all_tokens[:trimmed].reshape(n_sequences, pack_len)
    packed_masks = all_masks[:trimmed].reshape(n_sequences, pack_len)
    del all_tokens, all_masks
    gc.collect()

    # Shuffle (same permutation for tokens and masks)
    print(f"  Shuffling {n_sequences:,} sequences...")
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_sequences)
    packed_tokens = packed_tokens[perm]
    packed_masks = packed_masks[perm]

    # Split train/val (95/5)
    n_val = max(1, n_sequences // 20)
    n_train = n_sequences - n_val

    # Write train shards (tokens + masks)
    seqs_per_shard = 100_000
    shard_idx = 0
    for start in range(0, n_train, seqs_per_shard):
        end = min(start + seqs_per_shard, n_train)
        # Token shard
        tok_path = V5_INSTRUCT / f"train_{shard_idx:04d}.bin"
        packed_tokens[start:end].tofile(str(tok_path))
        # Mask shard
        mask_path = V5_INSTRUCT / f"train_mask_{shard_idx:04d}.bin"
        packed_masks[start:end].tofile(str(mask_path))
        print(f"  train_{shard_idx:04d}: {end - start:,} seqs (tokens + masks)")
        shard_idx += 1

    # Write val shards
    val_tok = V5_INSTRUCT / "val_0000.bin"
    packed_tokens[n_train:].tofile(str(val_tok))
    val_mask = V5_INSTRUCT / "val_mask_0000.bin"
    packed_masks[n_train:].tofile(str(val_mask))
    print(f"  val_0000: {n_val:,} seqs (tokens + masks)")

    # Metadata
    meta = {
        "total_tokens": int(total_tokens),
        "seq_len": SEQ_LEN,
        "pack_len": pack_len,
        "n_train_sequences": int(n_train),
        "n_val_sequences": int(n_val),
        "n_train_shards": shard_idx,
        "dtype": "uint16",
        "has_loss_mask": True,
    }
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)

    # Stats on mask coverage
    total_masked = int(packed_masks[:n_train].sum())
    total_positions = n_train * pack_len
    pct = total_masked / total_positions * 100
    print(f"\nV5 instruct shards: {n_train:,} train + {n_val:,} val sequences")
    print(f"  Loss mask coverage: {pct:.1f}% of tokens are assistant responses")

    del packed_tokens, packed_masks
    gc.collect()


# ═════════════════════════════════════════════════════════════════════════
# Stage 5: Finetune with loss masking
# ═════════════════════════════════════════════════════════════════════════
def stage_finetune():
    banner("1.0 Stage 5: Finetune with loss masking (20k steps)")

    finetune_ckpt = CKPT_DIR / "finetune_1.0_final.pt"
    if finetune_ckpt.exists():
        print(f"[skip] 1.0 finetune checkpoint exists: {finetune_ckpt}")
        return

    # Find pretrain checkpoint to finetune from
    pretrain_ckpt = CKPT_DIR / "pretrain_1.0_final.pt"
    if not pretrain_ckpt.exists():
        print("WARNING: No 1.0 pretrain checkpoint. Using V4 pretrain instead.")
        pretrain_ckpt = CKPT_DIR / "pretrain_v4_final.pt"
        if not pretrain_ckpt.exists():
            print("ERROR: No pretrain checkpoint found!")
            sys.exit(1)

    # Check for finetune resume checkpoints
    ft_ckpts = sorted(CKPT_DIR.glob("1.0_ft_step_*.pt"))
    if ft_ckpts:
        resume = ft_ckpts[-1]
        print(f"Resuming 1.0 finetune from: {resume}")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(FINETUNE_CFG),
               "--resume", str(resume)]
    else:
        print(f"Starting 1.0 finetune from: {pretrain_ckpt}")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(FINETUNE_CFG),
               "--finetune", str(pretrain_ckpt)]

    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print(f"WARNING: Finetune exited with code {result.returncode}")

    final = CKPT_DIR / "final.pt"
    if final.exists():
        shutil.copy2(str(final), str(finetune_ckpt))
        print(f"\nV5 finetune checkpoint saved: {finetune_ckpt}")
    else:
        ft_ckpts = sorted(CKPT_DIR.glob("1.0_ft_step_*.pt"))
        if ft_ckpts:
            latest = ft_ckpts[-1]
            shutil.copy2(str(latest), str(finetune_ckpt))
            print(f"\nUsing latest checkpoint: {latest} → {finetune_ckpt}")


# ═════════════════════════════════════════════════════════════════════════
# Stage 6: Test
# ═════════════════════════════════════════════════════════════════════════
def stage_test():
    banner("1.0 Stage 6: Quick test")

    # Find the best checkpoint
    for ckpt_name in ["finetune_1.0_final.pt", "pretrain_1.0_final.pt"]:
        ckpt = CKPT_DIR / ckpt_name
        if ckpt.exists():
            break
    else:
        print("ERROR: No 1.0 checkpoint found for testing!")
        return

    print(f"Testing with: {ckpt}")
    print(f"Config: {FINETUNE_CFG}")
    print()

    test_prompts = [
        "What color is the sky?",
        "What is 2 + 2?",
        "Who was the first president of the United States?",
        "Where is Paris?",
        "Write two sentences about why a PC is good.",
    ]

    # Use the inference module
    try:
        import torch
        import sentencepiece as spm
        from src.model.config import ModelConfig
        from src.model.transformer import Transformer
        from src.train.utils import load_config, load_checkpoint

        cfg = load_config(str(FINETUNE_CFG))
        model_cfg = ModelConfig.from_dict(cfg["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Transformer(model_cfg).to(device)
        load_checkpoint(str(ckpt), model)
        model.eval()

        sp = spm.SentencePieceProcessor()
        sp.load(str(TOKENIZER_PATH))

        print(f"Model loaded ({model.count_parameters():,} params)")
        print()

        for prompt in test_prompts:
            full = f"User: {prompt}\nAssistant:"
            tokens = sp.encode(full, out_type=int)
            x = torch.tensor([tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                for _ in range(200):
                    logits = model(x[:, -model_cfg.max_seq_len:])
                    next_logits = logits[:, -1, :] / 0.7
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    if next_token.item() == sp.eos_id():
                        break
                    x = torch.cat([x, next_token], dim=1)

            response = sp.decode(x[0].tolist()[len(tokens):])
            print(f"Q: {prompt}")
            print(f"A: {response.strip()}")
            print()

    except Exception as e:
        print(f"Auto-test failed: {e}")
        print(f"\nManual test command:")
        print(f"  python -m src.inference.chat --checkpoint {ckpt} --config {FINETUNE_CFG}")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="1386.ai 1.0 training pipeline")
    parser.add_argument("--stage", choices=["cleanup", "download", "shards", "pretrain",
                                            "instruct", "finetune", "test"],
                        help="Run a specific stage")
    args = parser.parse_args()

    print("=" * 64)
    print("  1386.ai 1.0 training pipeline")
    print("=" * 64)

    stages = {
        "cleanup": stage_cleanup,
        "download": stage_download,
        "shards": stage_build_pretrain_shards,
        "pretrain": stage_pretrain,
        "instruct": stage_build_instruct_shards,
        "finetune": stage_finetune,
        "test": stage_test,
    }

    if args.stage:
        stages[args.stage]()
    else:
        for name, fn in stages.items():
            fn()

    print("\n" + "=" * 64)
    print("  1.0 training complete!")
    print("  Test your model:")
    print(f"    python -m src.inference.chat --checkpoint checkpoints/finetune_1.0_final.pt --config {FINETUNE_CFG}")
    print("=" * 64)


if __name__ == "__main__":
    main()
