#!/usr/bin/env python3
# next gen training pipeline

import argparse
import gc
import hashlib
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
V6_RAW         = ROOT / "data" / "raw_v6"
V6_PRETRAIN    = ROOT / "data" / "shards_1.1"
V6_INSTRUCT    = ROOT / "data" / "instruct_shards_1.1"
TOKENIZER_PATH = ROOT / "data" / "tokenizer_v4.model"
CKPT_DIR       = ROOT / "checkpoints"
LOG_DIR        = ROOT / "logs"

PRETRAIN_CFG   = ROOT / "configs" / "pretrain_1.1.yaml"
FINETUNE_CFG   = ROOT / "configs" / "finetune_1.1.yaml"

VOCAB_SIZE = 32000
SEQ_LEN = 1024

# Data targets
FINEWEB_TARGET_CHARS = 25_000_000_000    # ~25GB text -> ~7B tokens
WIKI_TARGET_CHARS    = 6_000_000_000     # ~6GB text -> ~2B tokens
STACKEX_TARGET_CHARS = 4_000_000_000     # ~4GB text -> ~1B tokens


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


class ParagraphDeduplicator:
    """Fast exact-hash deduplication at paragraph level."""

    def __init__(self):
        self.seen_hashes = set()
        self.total_paras = 0
        self.deduped_paras = 0

    def dedup_text(self, text):
        """Remove duplicate paragraphs from text. Returns cleaned text."""
        paragraphs = text.split("\n\n")
        kept = []
        for para in paragraphs:
            para_stripped = para.strip()
            if len(para_stripped) < 50:
                kept.append(para)
                continue

            self.total_paras += 1
            h = hashlib.md5(para_stripped.lower().encode()).digest()
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                kept.append(para)
            else:
                self.deduped_paras += 1

        return "\n\n".join(kept)

    def stats(self):
        pct = (self.deduped_paras / max(1, self.total_paras)) * 100
        return (f"  Dedup: {self.deduped_paras:,}/{self.total_paras:,} "
                f"paragraphs removed ({pct:.1f}%)")


def quality_filter(text):
    """Filter out low-quality documents."""
    if len(text) < 200:
        return False
    sentences = text.split(".")
    if len(sentences) > 2:
        avg_sent_len = len(text) / len(sentences)
        if avg_sent_len < 15:
            return False
    if text.count("http") > 10:
        return False
    lines = text.strip().split("\n")
    if len(lines) > 5:
        unique_lines = set(l.strip() for l in lines if len(l.strip()) > 20)
        if len(unique_lines) < len(lines) * 0.5:
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════
# Stage 0: Clean up old intermediate checkpoints
# ═════════════════════════════════════════════════════════════════════════
def stage_cleanup():
    banner("1.1 Stage 0: Clean up old intermediate checkpoints")

    freed = 0

    for pattern in ["v5_step_*.pt", "v5_ft_step_*.pt"]:
        ckpts = list(CKPT_DIR.glob(pattern))
        if ckpts:
            for ckpt in ckpts:
                size = ckpt.stat().st_size
                ckpt.unlink()
                freed += size
            print(f"  Deleted {len(ckpts)} intermediate checkpoints: {pattern}")

    old_final = CKPT_DIR / "final.pt"
    if old_final.exists():
        freed += old_final.stat().st_size
        old_final.unlink()
        print(f"  Deleted old final.pt")

    if freed > 0:
        print(f"\n  Freed {freed / 1e9:.1f} GB of disk space")
    else:
        print("  Nothing to clean up")

    for name in ["finetune_v5_final.pt", "pretrain_v5_final.pt"]:
        p = CKPT_DIR / name
        if p.exists():
            print(f"  [kept] {name} ({p.stat().st_size / 1e9:.1f} GB)")


# ═════════════════════════════════════════════════════════════════════════
# Stage 1: Download high-quality data with deduplication
# ═════════════════════════════════════════════════════════════════════════
def stage_download():
    banner("1.1 Stage 1: Download high-quality deduped data")

    V6_RAW.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)

    dedup = ParagraphDeduplicator()

    # ── 1a: FineWeb-Edu (score >= 3 only = top quality) ──────────────
    fineweb_path = V6_RAW / "fineweb_edu_hq.txt"
    if fineweb_path.exists() and fineweb_path.stat().st_size >= FINEWEB_TARGET_CHARS * 0.7:
        print(f"[skip] FineWeb-Edu HQ exists ({fineweb_path.stat().st_size / 1e9:.2f} GB)")
    else:
        print(f"Downloading FineWeb-Edu (score >= 3, target: {FINEWEB_TARGET_CHARS / 1e9:.0f} GB)...")
        print("  Only keeping high-quality educational documents.")
        print("  This filters out ~40% of low-quality content.\n")

        total_chars = 0
        doc_count = 0
        skipped = 0
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
                    score = example.get("score", 0)
                    if score is not None and score < 3.0:
                        skipped += 1
                        continue

                    text = example["text"].strip()
                    if not quality_filter(text):
                        skipped += 1
                        continue

                    text = dedup.dedup_text(text)
                    if len(text.strip()) < 200:
                        continue

                    f.write(text)
                    f.write("\n\n")
                    total_chars += len(text) + 2
                    doc_count += 1

                    if doc_count % 50000 == 0:
                        speed = total_chars / (time.time() - t0 + 0.1) / 1e6
                        print(f"  {doc_count:,} docs | {total_chars / 1e9:.2f} GB | "
                              f"skipped {skipped:,} | {speed:.1f} MB/s")

                    if total_chars >= FINEWEB_TARGET_CHARS:
                        break

            dt = time.time() - t0
            print(f"\n  FineWeb-Edu HQ: {doc_count:,} docs, "
                  f"{fineweb_path.stat().st_size / 1e9:.2f} GB, "
                  f"skipped {skipped:,} low-quality | {elapsed_str(dt)}")
            print(dedup.stats())

        except Exception as e:
            print(f"  ERROR downloading FineWeb-Edu: {e}")
            print("  Will continue with other data sources.")

    # ── 1b: Wikipedia (clean, factual text) ──────────────────────────
    wiki_path = V6_RAW / "wikipedia_clean.txt"
    if wiki_path.exists() and wiki_path.stat().st_size >= WIKI_TARGET_CHARS * 0.7:
        print(f"\n[skip] Wikipedia exists ({wiki_path.stat().st_size / 1e9:.2f} GB)")
    else:
        print(f"\nDownloading Wikipedia (target: {WIKI_TARGET_CHARS / 1e9:.0f} GB)...")
        print("  Clean encyclopedic text for factual knowledge.\n")

        total_chars = 0
        doc_count = 0
        t0 = time.time()

        try:
            ds = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

            with open(wiki_path, "w", encoding="utf-8") as f:
                for example in ds:
                    text = example["text"].strip()
                    if len(text) < 300:
                        continue

                    text = dedup.dedup_text(text)
                    if len(text.strip()) < 200:
                        continue

                    f.write(text)
                    f.write("\n\n")
                    total_chars += len(text) + 2
                    doc_count += 1

                    if doc_count % 50000 == 0:
                        speed = total_chars / (time.time() - t0 + 0.1) / 1e6
                        print(f"  {doc_count:,} docs | {total_chars / 1e9:.2f} GB | "
                              f"{speed:.1f} MB/s")

                    if total_chars >= WIKI_TARGET_CHARS:
                        break

            dt = time.time() - t0
            print(f"\n  Wikipedia: {doc_count:,} docs, "
                  f"{wiki_path.stat().st_size / 1e9:.2f} GB | {elapsed_str(dt)}")
            print(dedup.stats())

        except Exception as e:
            print(f"  ERROR downloading Wikipedia: {e}")
            print("  Will continue with available data.")

    # ── 1c: StackExchange (reasoning + Q&A) ──────────────────────────
    stackex_path = V6_RAW / "stackexchange_clean.txt"
    if stackex_path.exists() and stackex_path.stat().st_size >= STACKEX_TARGET_CHARS * 0.5:
        print(f"\n[skip] StackExchange exists ({stackex_path.stat().st_size / 1e9:.2f} GB)")
    else:
        print(f"\nDownloading StackExchange (target: {STACKEX_TARGET_CHARS / 1e9:.0f} GB)...")
        print("  High-quality Q&A pairs for reasoning ability.\n")

        total_chars = 0
        doc_count = 0
        skipped = 0
        t0 = time.time()

        try:
            ds = load_dataset(
                "HuggingFaceTB/stack-exchange-preferences",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

            with open(stackex_path, "w", encoding="utf-8") as f:
                for example in ds:
                    question = example.get("question", "")
                    answers = example.get("answers", [])
                    if not answers:
                        continue

                    best_answer = None
                    best_score = -1
                    for ans in answers:
                        score = ans.get("pm_score", ans.get("score", 0))
                        if score is not None and score > best_score:
                            best_score = score
                            best_answer = ans.get("text", "")

                    if not best_answer or len(best_answer) < 100:
                        skipped += 1
                        continue

                    text = f"Question: {question}\n\nAnswer: {best_answer}"
                    if not quality_filter(text):
                        skipped += 1
                        continue

                    text = dedup.dedup_text(text)
                    f.write(text)
                    f.write("\n\n")
                    total_chars += len(text) + 2
                    doc_count += 1

                    if doc_count % 50000 == 0:
                        speed = total_chars / (time.time() - t0 + 0.1) / 1e6
                        print(f"  {doc_count:,} docs | {total_chars / 1e9:.2f} GB | "
                              f"skipped {skipped:,} | {speed:.1f} MB/s")

                    if total_chars >= STACKEX_TARGET_CHARS:
                        break

            dt = time.time() - t0
            print(f"\n  StackExchange: {doc_count:,} docs, "
                  f"{stackex_path.stat().st_size / 1e9:.2f} GB | {elapsed_str(dt)}")

        except Exception as e:
            print(f"  ERROR downloading StackExchange: {e}")
            print("  Will continue with available data.")

    # Final stats
    print(f"\n{'─' * 40}")
    print("Data download summary:")
    total_size = 0
    for name, path in [("FineWeb-Edu HQ", fineweb_path),
                        ("Wikipedia", wiki_path),
                        ("StackExchange", stackex_path)]:
        if path.exists():
            size = path.stat().st_size
            total_size += size
            print(f"  {name}: {size / 1e9:.2f} GB")
        else:
            print(f"  {name}: NOT DOWNLOADED")
    print(f"  Total: {total_size / 1e9:.2f} GB")
    print(dedup.stats())


# ═════════════════════════════════════════════════════════════════════════
# Stage 2: Build pretrain shards
# ═════════════════════════════════════════════════════════════════════════
def stage_build_pretrain_shards():
    banner("1.1 Stage 2: Build pretrain shards (~10B tokens)")

    meta_path = V6_PRETRAIN / "meta.yaml"
    if meta_path.exists():
        print(f"[skip] 1.1 pretrain shards exist: {V6_PRETRAIN}")
        return

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    eos_id = sp.eos_id()

    # Collect all corpus files
    corpus_files = []

    for name, filename in [("FineWeb-Edu HQ", "fineweb_edu_hq.txt"),
                            ("Wikipedia", "wikipedia_clean.txt"),
                            ("StackExchange", "stackexchange_clean.txt")]:
        path = V6_RAW / filename
        if path.exists():
            corpus_files.append((name, path))

    v4_raw = ROOT / "data" / "raw_v4"
    for name, filename in [("OpenWebText+Wiki (V4)", "pretrain_corpus.txt"),
                            ("FineWeb-Edu", "fineweb_edu_corpus.txt")]:
        path = v4_raw / filename
        if path.exists():
            corpus_files.append((name, path))

    if not corpus_files:
        print("ERROR: No corpus files found. Run stage 1 first.")
        sys.exit(1)

    total_size = sum(f.stat().st_size for _, f in corpus_files)
    print(f"Corpus files ({total_size / 1e9:.1f} GB total):")
    for name, path in corpus_files:
        print(f"  {name}: {path.stat().st_size / 1e9:.2f} GB")

    pack_len = SEQ_LEN + 1
    V6_PRETRAIN.mkdir(parents=True, exist_ok=True)

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
                    if len(token_chunks) % 1000 == 0:
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

    print("  Packing into sequences...")
    all_tokens = np.concatenate(token_chunks)
    del token_chunks
    gc.collect()

    n_sequences = len(all_tokens) // pack_len
    trimmed = n_sequences * pack_len
    packed = all_tokens[:trimmed].reshape(n_sequences, pack_len)
    del all_tokens
    gc.collect()

    print(f"  Shuffling {n_sequences:,} sequences...")
    rng = np.random.default_rng(42)
    rng.shuffle(packed)

    n_val = max(1, n_sequences // 33)
    n_train = n_sequences - n_val

    seqs_per_shard = 100_000
    shard_idx = 0
    for start in range(0, n_train, seqs_per_shard):
        end = min(start + seqs_per_shard, n_train)
        path = V6_PRETRAIN / f"train_{shard_idx:04d}.bin"
        packed[start:end].tofile(str(path))
        if shard_idx % 10 == 0:
            print(f"  train_{shard_idx:04d}: {end - start:,} seqs")
        shard_idx += 1

    print(f"  ... {shard_idx} train shards total")

    val_path = V6_PRETRAIN / "val_0000.bin"
    packed[n_train:].tofile(str(val_path))
    print(f"  val_0000.bin: {n_val:,} seqs")

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
    print(f"\nV6 pretrain shards: {n_train:,} train + {n_val:,} val sequences")
    print(f"  Total tokens: {total_tokens / 1e9:.3f}B")


# ═════════════════════════════════════════════════════════════════════════
# Stage 3: Pretrain 500M model from scratch
# ═════════════════════════════════════════════════════════════════════════
def stage_pretrain():
    banner("1.1 Stage 3: Pretrain 500M model (200k steps, ~5 days)")

    pretrain_ckpt = CKPT_DIR / "pretrain_1.1_final.pt"
    if pretrain_ckpt.exists():
        print(f"[skip] 1.1 pretrain checkpoint exists: {pretrain_ckpt}")
        return

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    v6_ckpts = sorted(CKPT_DIR.glob("1.1_step_*.pt"))
    if v6_ckpts:
        resume_from = v6_ckpts[-1]
        print(f"Resuming 1.1 pretraining from: {resume_from}")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(PRETRAIN_CFG),
               "--resume", str(resume_from)]
    else:
        print("Starting 1.1 pretraining from scratch (500M params)")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(PRETRAIN_CFG)]

    print(f"  Config: {PRETRAIN_CFG}")
    print(f"  Target: 200,000 steps (~5 days on RTX 5080)")
    print(f"  Checkpoints saved every 10000 steps")
    print(f"  You can stop and resume anytime.")
    print()

    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print(f"WARNING: Training exited with code {result.returncode}")

    final = CKPT_DIR / "final.pt"
    if final.exists():
        shutil.copy2(str(final), str(pretrain_ckpt))
        print(f"\nV6 pretrain checkpoint saved: {pretrain_ckpt}")
    else:
        v6_ckpts = sorted(CKPT_DIR.glob("1.1_step_*.pt"))
        if v6_ckpts:
            latest = v6_ckpts[-1]
            shutil.copy2(str(latest), str(pretrain_ckpt))
            print(f"\nUsing latest 1.1 checkpoint: {latest} -> {pretrain_ckpt}")
        else:
            print("ERROR: No checkpoint found after pretraining!")
            sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════
# Stage 4: Download + build high-quality multi-turn instruct data
# ═════════════════════════════════════════════════════════════════════════
def _download_instruct_data():
    """Download high-quality instruction datasets."""
    from datasets import load_dataset

    V6_RAW.mkdir(parents=True, exist_ok=True)
    instruct_path = V6_RAW / "instruct_v6.jsonl"

    if instruct_path.exists() and instruct_path.stat().st_size > 500_000_000:
        print(f"[skip] 1.1 instruct data exists ({instruct_path.stat().st_size / 1e9:.2f} GB)")
        return instruct_path

    print("Downloading high-quality instruction datasets...\n")
    t0 = time.time()
    total_convs = 0

    with open(instruct_path, "w", encoding="utf-8") as out:

        # ── Source 1: SlimOrca (GPT-4 quality responses) ─────────────
        print("  [1/3] SlimOrca (GPT-4 generated, ~500k conversations)...")
        try:
            ds = load_dataset("Open-Orca/SlimOrca", split="train",
                              streaming=True, trust_remote_code=True)
            count = 0
            for example in ds:
                convs = example.get("conversations", [])
                if not convs or len(convs) < 2:
                    continue

                # Format as multi-turn: User/Assistant pairs
                turns = []
                for msg in convs:
                    role = msg.get("from", "")
                    value = msg.get("value", "").strip()
                    if not value:
                        continue
                    if role in ("human", "user"):
                        turns.append(f"User: {value}")
                    elif role in ("gpt", "assistant"):
                        turns.append(f"Assistant: {value}")
                    # Skip system messages

                if len(turns) >= 2:
                    text = "\n".join(turns)
                    if len(text) > 50:
                        out.write(json.dumps({"text": text, "source": "slimorca"}) + "\n")
                        count += 1
                        total_convs += 1

                if count % 50000 == 0 and count > 0:
                    print(f"    {count:,} conversations...")
                if count >= 500000:
                    break

            print(f"    SlimOrca: {count:,} conversations")
        except Exception as e:
            print(f"    ERROR: {e}")

        # ── Source 2: UltraChat (multi-turn conversations) ───────────
        print("  [2/3] UltraChat 200k (multi-turn conversations)...")
        try:
            ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                              split="train_sft", streaming=True,
                              trust_remote_code=True)
            count = 0
            for example in ds:
                messages = example.get("messages", [])
                if len(messages) < 2:
                    continue

                turns = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "").strip()
                    if not content:
                        continue
                    if role == "user":
                        turns.append(f"User: {content}")
                    elif role == "assistant":
                        turns.append(f"Assistant: {content}")

                if len(turns) >= 2:
                    text = "\n".join(turns)
                    if len(text) > 50:
                        out.write(json.dumps({"text": text, "source": "ultrachat"}) + "\n")
                        count += 1
                        total_convs += 1

                if count % 50000 == 0 and count > 0:
                    print(f"    {count:,} conversations...")
                if count >= 200000:
                    break

            print(f"    UltraChat: {count:,} conversations")
        except Exception as e:
            print(f"    ERROR: {e}")

        # ── Source 3: Existing V4 instruct data ──────────────────────
        v4_instruct = ROOT / "data" / "raw_v4" / "instruct_corpus.jsonl"
        if v4_instruct.exists():
            print("  [3/3] Existing V4 instruct data...")
            count = 0
            with open(v4_instruct, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        text = data["text"]
                        if len(text) > 50:
                            out.write(json.dumps({"text": text, "source": "v4"}) + "\n")
                            count += 1
                            total_convs += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

            print(f"    V4 instruct: {count:,} conversations")

    dt = time.time() - t0
    print(f"\n  Total: {total_convs:,} conversations, "
          f"{instruct_path.stat().st_size / 1e9:.2f} GB | {elapsed_str(dt)}")
    return instruct_path


def _build_multiturn_mask(text, sp):
    """Build loss mask for multi-turn conversation.

    Mask = 1 for ALL assistant response tokens, 0 for all user/system tokens.
    Handles multiple turns: User/Assistant/User/Assistant/...
    """
    tokens = sp.encode(text, out_type=int)
    if len(tokens) < 4:
        return None, None

    mask = np.zeros(len(tokens), dtype=np.uint8)

    # Find all "Assistant:" positions in the text and mask their tokens
    # We process the text to find each assistant segment
    pos = 0
    while pos < len(text):
        # Find next "Assistant:" marker
        asst_start = text.find("Assistant:", pos)
        if asst_start < 0:
            break

        # Find where this assistant response ends (next "User:" or end of text)
        next_user = text.find("\nUser:", asst_start + 10)
        if next_user < 0:
            asst_end = len(text)
        else:
            asst_end = next_user

        # Get token indices for this assistant segment
        prefix_tokens = sp.encode(text[:asst_start], out_type=int)
        segment_tokens = sp.encode(text[:asst_end], out_type=int)

        start_idx = len(prefix_tokens)
        end_idx = len(segment_tokens)
        mask[start_idx:end_idx] = 1

        pos = asst_end + 1

    return tokens, mask


def stage_build_instruct_shards():
    banner("1.1 Stage 4: Build high-quality multi-turn instruct shards")

    meta_path = V6_INSTRUCT / "meta.yaml"
    if meta_path.exists():
        print(f"[skip] 1.1 instruct shards exist: {V6_INSTRUCT}")
        return

    # Download high-quality instruct data if needed
    instruct_path = _download_instruct_data()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    eos_id = sp.eos_id()

    pack_len = SEQ_LEN + 1
    V6_INSTRUCT.mkdir(parents=True, exist_ok=True)

    print("\nTokenizing multi-turn instruction data with loss masks...")
    t0 = time.time()

    all_tokens = []
    all_masks = []
    total_tokens = 0
    n_convs = 0
    n_multiturn = 0

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

            # Build multi-turn aware loss mask
            tokens, mask = _build_multiturn_mask(text, sp)
            if tokens is None:
                continue

            # Count multi-turn conversations
            if text.count("User:") > 1:
                n_multiturn += 1

            # Add EOS (should be predicted = mask 1)
            tokens.append(eos_id)
            mask = np.append(mask, 1)

            all_tokens.extend(tokens)
            all_masks.extend(mask)
            total_tokens += len(tokens)
            n_convs += 1

            if n_convs % 100000 == 0:
                print(f"  {n_convs:,} conversations | "
                      f"{total_tokens:,} tokens ({total_tokens / 1e6:.1f}M) | "
                      f"multi-turn: {n_multiturn:,}")

    dt = time.time() - t0
    print(f"  Tokenized {n_convs:,} conversations -> "
          f"{total_tokens:,} tokens ({total_tokens / 1e6:.1f}M) "
          f"in {elapsed_str(dt)}")
    print(f"  Multi-turn conversations: {n_multiturn:,} "
          f"({n_multiturn / max(1, n_convs) * 100:.1f}%)")

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

    print(f"  Shuffling {n_sequences:,} sequences...")
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_sequences)
    packed_tokens = packed_tokens[perm]
    packed_masks = packed_masks[perm]

    n_val = max(1, n_sequences // 20)
    n_train = n_sequences - n_val

    seqs_per_shard = 100_000
    shard_idx = 0
    for start in range(0, n_train, seqs_per_shard):
        end = min(start + seqs_per_shard, n_train)
        tok_path = V6_INSTRUCT / f"train_{shard_idx:04d}.bin"
        packed_tokens[start:end].tofile(str(tok_path))
        mask_path = V6_INSTRUCT / f"train_mask_{shard_idx:04d}.bin"
        packed_masks[start:end].tofile(str(mask_path))
        print(f"  train_{shard_idx:04d}: {end - start:,} seqs (tokens + masks)")
        shard_idx += 1

    val_tok = V6_INSTRUCT / "val_0000.bin"
    packed_tokens[n_train:].tofile(str(val_tok))
    val_mask = V6_INSTRUCT / "val_mask_0000.bin"
    packed_masks[n_train:].tofile(str(val_mask))
    print(f"  val_0000: {n_val:,} seqs (tokens + masks)")

    meta = {
        "total_tokens": int(total_tokens),
        "seq_len": SEQ_LEN,
        "pack_len": pack_len,
        "n_train_sequences": int(n_train),
        "n_val_sequences": int(n_val),
        "n_train_shards": shard_idx,
        "dtype": "uint16",
        "has_loss_mask": True,
        "multiturn": True,
    }
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)

    total_masked = int(packed_masks[:n_train].sum())
    total_positions = n_train * pack_len
    pct = total_masked / total_positions * 100
    print(f"\nV6 instruct shards: {n_train:,} train + {n_val:,} val sequences")
    print(f"  Loss mask coverage: {pct:.1f}% of tokens are assistant responses")
    print(f"  Multi-turn conversations: {n_multiturn:,}")

    del packed_tokens, packed_masks
    gc.collect()


# ═════════════════════════════════════════════════════════════════════════
# Stage 5: Finetune with loss masking
# ═════════════════════════════════════════════════════════════════════════
def stage_finetune():
    banner("1.1 Stage 5: Finetune with loss masking (30k steps)")

    finetune_ckpt = CKPT_DIR / "finetune_1.1_final.pt"
    if finetune_ckpt.exists():
        print(f"[skip] 1.1 finetune checkpoint exists: {finetune_ckpt}")
        return

    pretrain_ckpt = CKPT_DIR / "pretrain_1.1_final.pt"
    if not pretrain_ckpt.exists():
        print("ERROR: No 1.1 pretrain checkpoint found!")
        sys.exit(1)

    ft_ckpts = sorted(CKPT_DIR.glob("1.1_ft_step_*.pt"))
    if ft_ckpts:
        resume = ft_ckpts[-1]
        print(f"Resuming 1.1 finetune from: {resume}")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(FINETUNE_CFG),
               "--resume", str(resume)]
    else:
        print(f"Starting 1.1 finetune from: {pretrain_ckpt}")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(FINETUNE_CFG),
               "--finetune", str(pretrain_ckpt)]

    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print(f"WARNING: Finetune exited with code {result.returncode}")

    final = CKPT_DIR / "final.pt"
    if final.exists():
        shutil.copy2(str(final), str(finetune_ckpt))
        print(f"\nV6 finetune checkpoint saved: {finetune_ckpt}")
    else:
        ft_ckpts = sorted(CKPT_DIR.glob("1.1_ft_step_*.pt"))
        if ft_ckpts:
            latest = ft_ckpts[-1]
            shutil.copy2(str(latest), str(finetune_ckpt))
            print(f"\nUsing latest checkpoint: {latest} -> {finetune_ckpt}")


# ═════════════════════════════════════════════════════════════════════════
# Stage 6: Test
# ═════════════════════════════════════════════════════════════════════════
def stage_test():
    banner("1.1 Stage 6: Quick test")

    for ckpt_name in ["finetune_1.1_final.pt", "pretrain_1.1_final.pt"]:
        ckpt = CKPT_DIR / ckpt_name
        if ckpt.exists():
            break
    else:
        print("ERROR: No 1.1 checkpoint found for testing!")
        return

    print(f"Testing with: {ckpt}")
    print(f"Config: {FINETUNE_CFG}")
    print()

    test_prompts = [
        "What color is the sky?",
        "What is 2 + 2?",
        "Who was the first president of the United States?",
        "Where is France located?",
        "What continent is Japan on?",
        "Who was Napoleon?",
        "Write two sentences about why exercise is good for you.",
        "What is the capital of Germany?",
    ]

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
                    next_logits = logits[:, -1, :] / 0.3
                    top_vals, top_idx = torch.topk(next_logits, 8)
                    probs = torch.softmax(top_vals, dim=-1)
                    chosen = torch.multinomial(probs, 1)
                    next_token = top_idx.gather(1, chosen)
                    if next_token.item() == sp.eos_id():
                        break
                    x = torch.cat([x, next_token], dim=1)

            response = sp.decode(x[0].tolist()[len(tokens):])
            for stop in ["\nUser:", "\nSystem:", "\nHuman:"]:
                if stop in response:
                    response = response[:response.index(stop)]
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
    parser = argparse.ArgumentParser(description="1386.ai next gen training pipeline")
    parser.add_argument("--stage", choices=["cleanup", "download", "shards", "pretrain",
                                            "instruct", "finetune", "test"],
                        help="Run a specific stage")
    args = parser.parse_args()

    print("=" * 64)
    print("  1386.ai next gen training pipeline")
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
    print("  1.1 training complete!")
    print("  Test your model:")
    print(f"    python -m src.inference.chat --checkpoint checkpoints/finetune_1.1_final.pt --config {FINETUNE_CFG}")
    print("  Or launch the web UI:")
    print("    python run.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
