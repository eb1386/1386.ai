#!/usr/bin/env python3
# 1.1 training pipeline

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# paths
RAW_DIR          = ROOT / "data" / "raw_1.1"
SCORED_DIR       = ROOT / "data" / "scored_1.1"
DEDUPED_DIR      = ROOT / "data" / "deduped_1.1"
PRETRAIN_SHARDS  = ROOT / "data" / "shards_1.1"
INSTRUCT_SHARDS  = ROOT / "data" / "instruct_shards_1.1"
TOKENIZER_OLD    = ROOT / "data" / "tokenizer_v4.model"
TOKENIZER_NEW    = ROOT / "data" / "tokenizer_1.1"
CKPT_DIR         = ROOT / "checkpoints"
LOG_DIR          = ROOT / "logs"

PRETRAIN_CFG     = ROOT / "configs" / "pretrain_1.1.yaml"
FINETUNE_CFG     = ROOT / "configs" / "finetune_1.1.yaml"

SEQ_LEN = 1024

# data targets (chars)
FINEWEB_TARGET    = 30_000_000_000
WIKI_TARGET       = 6_000_000_000
STACKEX_TARGET    = 4_000_000_000
CODE_TARGET       = 8_000_000_000
ARXIV_TARGET      = 3_000_000_000

QUALITY_MIN_SCORE = 0.55


def banner(msg):
    print(f"\n{'=' * 64}")
    print(f"  {msg}")
    print(f"{'=' * 64}\n")


def elapsed_str(seconds):
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ── stage 0: cleanup ────────────────────────────────────────────────
def stage_cleanup():
    banner("Stage 0: Cleanup")

    freed = 0
    if CKPT_DIR.exists():
        for pattern in ["1.1_step_*.pt", "1.1_ft_step_*.pt"]:
            ckpts = list(CKPT_DIR.glob(pattern))
            if ckpts:
                for ckpt in ckpts:
                    freed += ckpt.stat().st_size
                    ckpt.unlink()
                print(f"  Deleted {len(ckpts)} checkpoints: {pattern}")

    if freed > 0:
        print(f"\n  Freed {freed / 1e9:.1f} GB")
    else:
        print("  Nothing to clean up")

    for name in ["finetune_1.1_final.pt", "pretrain_1.1_final.pt",
                 "finetune_1.0_final.pt", "pretrain_1.0_final.pt"]:
        p = CKPT_DIR / name
        if p.exists():
            print(f"  [kept] {name} ({p.stat().st_size / 1e9:.1f} GB)")


# ── stage 1: download ───────────────────────────────────────────────
def stage_download():
    banner("Stage 1: Download training data")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    _download_fineweb(load_dataset)
    _download_wikipedia(load_dataset)
    _download_stackexchange(load_dataset)
    _download_code(load_dataset)
    _download_arxiv(load_dataset)

    print(f"\n{'─' * 40}")
    print("Download summary:")
    total = 0
    for name in ["fineweb_edu_hq.txt", "wikipedia_clean.txt",
                 "stackexchange_clean.txt", "code_clean.txt", "arxiv_clean.txt"]:
        p = RAW_DIR / name
        if p.exists():
            size = p.stat().st_size
            total += size
            print(f"  {name:30s} {size / 1e9:.2f} GB")
    print(f"  {'Total':30s} {total / 1e9:.2f} GB")


def _download_fineweb(load_dataset):
    path = RAW_DIR / "fineweb_edu_hq.txt"
    if path.exists() and path.stat().st_size >= FINEWEB_TARGET * 0.7:
        print(f"[skip] FineWeb-Edu HQ ({path.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"Downloading FineWeb-Edu (score >= 3, target: {FINEWEB_TARGET / 1e9:.0f} GB)...")
    total_chars = 0
    doc_count = 0
    skipped = 0
    t0 = time.time()

    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True, trust_remote_code=True)

        with open(path, "w", encoding="utf-8") as f:
            for example in ds:
                score = example.get("score", 0)
                if score is not None and score < 3.0:
                    skipped += 1
                    continue
                text = example["text"].strip()
                if len(text) < 200:
                    skipped += 1
                    continue
                f.write(text + "\n\n")
                total_chars += len(text) + 2
                doc_count += 1
                if doc_count % 100000 == 0:
                    speed = total_chars / (time.time() - t0 + 0.1) / 1e6
                    print(f"  {doc_count:,} docs | {total_chars / 1e9:.2f} GB | "
                          f"skipped {skipped:,} | {speed:.1f} MB/s")
                if total_chars >= FINEWEB_TARGET:
                    break

        print(f"  FineWeb-Edu: {doc_count:,} docs, {path.stat().st_size / 1e9:.2f} GB "
              f"({elapsed_str(time.time() - t0)})")
    except Exception as e:
        print(f"  ERROR: {e}")


def _download_wikipedia(load_dataset):
    path = RAW_DIR / "wikipedia_clean.txt"
    if path.exists() and path.stat().st_size >= WIKI_TARGET * 0.7:
        print(f"[skip] Wikipedia ({path.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"Downloading Wikipedia (target: {WIKI_TARGET / 1e9:.0f} GB)...")
    total_chars = 0
    doc_count = 0
    t0 = time.time()

    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True, trust_remote_code=True)

        with open(path, "w", encoding="utf-8") as f:
            for example in ds:
                text = example["text"].strip()
                if len(text) < 300:
                    continue
                f.write(text + "\n\n")
                total_chars += len(text) + 2
                doc_count += 1
                if doc_count % 100000 == 0:
                    print(f"  {doc_count:,} docs | {total_chars / 1e9:.2f} GB")
                if total_chars >= WIKI_TARGET:
                    break

        print(f"  Wikipedia: {doc_count:,} docs, {path.stat().st_size / 1e9:.2f} GB "
              f"({elapsed_str(time.time() - t0)})")
    except Exception as e:
        print(f"  ERROR: {e}")


def _download_stackexchange(load_dataset):
    path = RAW_DIR / "stackexchange_clean.txt"
    if path.exists() and path.stat().st_size >= STACKEX_TARGET * 0.5:
        print(f"[skip] StackExchange ({path.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"Downloading StackExchange (target: {STACKEX_TARGET / 1e9:.0f} GB)...")
    total_chars = 0
    doc_count = 0
    skipped = 0
    t0 = time.time()

    try:
        ds = load_dataset("HuggingFaceTB/stack-exchange-preferences",
                          split="train", streaming=True, trust_remote_code=True)

        with open(path, "w", encoding="utf-8") as f:
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
                if best_score < 1:
                    skipped += 1
                    continue

                text = f"Question: {question}\n\nAnswer: {best_answer}"
                f.write(text + "\n\n")
                total_chars += len(text) + 2
                doc_count += 1
                if doc_count % 100000 == 0:
                    print(f"  {doc_count:,} docs | {total_chars / 1e9:.2f} GB")
                if total_chars >= STACKEX_TARGET:
                    break

        print(f"  StackExchange: {doc_count:,} docs, {path.stat().st_size / 1e9:.2f} GB "
              f"({elapsed_str(time.time() - t0)})")
    except Exception as e:
        print(f"  ERROR: {e}")


def _download_code(load_dataset):
    path = RAW_DIR / "code_clean.txt"
    if path.exists() and path.stat().st_size >= CODE_TARGET * 0.5:
        print(f"[skip] Code ({path.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"Downloading code (target: {CODE_TARGET / 1e9:.0f} GB)...")
    total_chars = 0
    doc_count = 0
    skipped = 0
    t0 = time.time()
    languages = ["python", "javascript", "typescript"]

    try:
        with open(path, "w", encoding="utf-8") as f:
            per_lang = CODE_TARGET // len(languages)
            for lang in languages:
                lang_chars = 0
                lang_docs = 0
                print(f"\n  Downloading {lang}...")

                try:
                    ds = load_dataset("bigcode/starcoderdata", data_dir=lang,
                                      split="train", streaming=True, trust_remote_code=True)
                    for example in ds:
                        content = example.get("content", "").strip()
                        if len(content) < 100:
                            skipped += 1
                            continue
                        lines = content.split("\n")
                        if len(lines) < 5:
                            skipped += 1
                            continue
                        code_lines = [l for l in lines if l.strip()
                                      and not l.strip().startswith("#")
                                      and not l.strip().startswith("//")
                                      and not l.strip().startswith("/*")]
                        if len(code_lines) < len(lines) * 0.3:
                            skipped += 1
                            continue
                        if len(content) > 50000:
                            skipped += 1
                            continue
                        max_line = max(len(l) for l in lines) if lines else 0
                        if max_line > 500:
                            skipped += 1
                            continue

                        f.write(content + "\n\n")
                        lang_chars += len(content) + 2
                        total_chars += len(content) + 2
                        lang_docs += 1
                        doc_count += 1
                        if lang_docs % 50000 == 0:
                            print(f"    {lang}: {lang_docs:,} files | {lang_chars / 1e9:.2f} GB")
                        if lang_chars >= per_lang:
                            break

                    print(f"    {lang}: {lang_docs:,} files, {lang_chars / 1e9:.2f} GB")
                except Exception as e:
                    print(f"    ERROR {lang}: {e}")

        print(f"\n  Code total: {doc_count:,} files, {path.stat().st_size / 1e9:.2f} GB "
              f"({elapsed_str(time.time() - t0)})")
    except Exception as e:
        print(f"  ERROR: {e}")


def _download_arxiv(load_dataset):
    path = RAW_DIR / "arxiv_clean.txt"
    if path.exists() and path.stat().st_size >= ARXIV_TARGET * 0.5:
        print(f"[skip] ArXiv ({path.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"Downloading ArXiv (target: {ARXIV_TARGET / 1e9:.0f} GB)...")
    total_chars = 0
    doc_count = 0
    t0 = time.time()

    try:
        ds = load_dataset("ccdv/arxiv-classification",
                          split="train", streaming=True, trust_remote_code=True)
        with open(path, "w", encoding="utf-8") as f:
            for example in ds:
                text = example.get("text", "").strip()
                if len(text) < 200:
                    continue
                f.write(text + "\n\n")
                total_chars += len(text) + 2
                doc_count += 1
                if doc_count % 50000 == 0:
                    print(f"  {doc_count:,} docs | {total_chars / 1e9:.2f} GB")
                if total_chars >= ARXIV_TARGET:
                    break

        print(f"  ArXiv: {doc_count:,} docs, {path.stat().st_size / 1e9:.2f} GB "
              f"({elapsed_str(time.time() - t0)})")
    except Exception as e:
        print(f"  ERROR: {e}")


# ── stage 2: quality scoring ────────────────────────────────────────
def stage_quality_score():
    banner("Stage 2: Quality scoring")

    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    from src.data.quality import filter_and_score

    source_files = [
        ("fineweb_edu_hq.txt", QUALITY_MIN_SCORE),
        ("wikipedia_clean.txt", 0.45),
        ("stackexchange_clean.txt", QUALITY_MIN_SCORE),
        ("code_clean.txt", 0.40),
        ("arxiv_clean.txt", 0.45),
    ]

    for filename, threshold in source_files:
        src = RAW_DIR / filename
        dst = SCORED_DIR / filename

        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {filename} already scored")
            continue
        if not src.exists():
            print(f"[skip] {filename} not downloaded")
            continue

        print(f"\nScoring {filename} (threshold: {threshold})...")
        t0 = time.time()
        total = 0
        kept = 0
        total_score = 0.0
        doc_buffer = []

        with open(src, "r", encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                doc_buffer.append(line.rstrip("\n"))
                if line.strip() == "" and len(doc_buffer) > 1:
                    text = "\n".join(doc_buffer).strip()
                    doc_buffer = []
                    if len(text) < 50:
                        continue
                    total += 1
                    passed, score = filter_and_score(text, threshold)
                    if passed:
                        fout.write(text + "\n\n")
                        kept += 1
                        total_score += score
                    if total % 100000 == 0:
                        avg = total_score / max(1, kept)
                        pct = kept / total * 100
                        print(f"  {total:,} scored | {kept:,} kept ({pct:.1f}%) | avg: {avg:.3f}")

            if doc_buffer:
                text = "\n".join(doc_buffer).strip()
                if len(text) >= 50:
                    total += 1
                    passed, score = filter_and_score(text, threshold)
                    if passed:
                        fout.write(text + "\n\n")
                        kept += 1
                        total_score += score

        dt = time.time() - t0
        avg = total_score / max(1, kept)
        pct = kept / max(1, total) * 100
        print(f"  {filename}: {kept:,}/{total:,} kept ({pct:.1f}%) | avg: {avg:.3f} | {elapsed_str(dt)}")


# ── stage 3: minhash dedup ──────────────────────────────────────────
def stage_minhash_dedup():
    banner("Stage 3: MinHash dedup")

    DEDUPED_DIR.mkdir(parents=True, exist_ok=True)
    from src.data.minhash import MinHashLSH

    lsh = MinHashLSH(n_hashes=128, n_bands=16, threshold=0.8, shingle_k=5)

    source_files = [
        "fineweb_edu_hq.txt",
        "wikipedia_clean.txt",
        "stackexchange_clean.txt",
        "code_clean.txt",
        "arxiv_clean.txt",
    ]

    doc_id = 0
    total_input = 0
    total_kept = 0

    for filename in source_files:
        src = SCORED_DIR / filename
        dst = DEDUPED_DIR / filename

        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {filename} already deduped")
            continue
        if not src.exists():
            print(f"[skip] {filename} not scored")
            continue

        print(f"\nDeduplicating {filename}...")
        t0 = time.time()
        file_input = 0
        file_kept = 0
        doc_buffer = []

        with open(src, "r", encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                doc_buffer.append(line.rstrip("\n"))
                if line.strip() == "" and len(doc_buffer) > 1:
                    text = "\n".join(doc_buffer).strip()
                    doc_buffer = []
                    if len(text) < 50:
                        continue
                    file_input += 1
                    total_input += 1
                    is_novel = lsh.insert(doc_id, text)
                    doc_id += 1
                    if is_novel:
                        fout.write(text + "\n\n")
                        file_kept += 1
                        total_kept += 1
                    if file_input % 100000 == 0:
                        pct = (file_input - file_kept) / max(1, file_input) * 100
                        stats = lsh.stats()
                        print(f"  {file_input:,} processed | {file_kept:,} kept | "
                              f"{pct:.1f}% dropped | index: {stats['memory_mb']:.0f} MB")

            if doc_buffer:
                text = "\n".join(doc_buffer).strip()
                if len(text) >= 50:
                    file_input += 1
                    total_input += 1
                    is_novel = lsh.insert(doc_id, text)
                    doc_id += 1
                    if is_novel:
                        fout.write(text + "\n\n")
                        file_kept += 1
                        total_kept += 1

        dt = time.time() - t0
        pct = (file_input - file_kept) / max(1, file_input) * 100
        print(f"  {filename}: {file_kept:,}/{file_input:,} kept ({pct:.1f}% dropped) | {elapsed_str(dt)}")

    overall_pct = (total_input - total_kept) / max(1, total_input) * 100
    stats = lsh.stats()
    print(f"\nDedup summary:")
    print(f"  {total_kept:,}/{total_input:,} kept ({overall_pct:.1f}% dropped)")
    print(f"  Index: {stats['memory_mb']:.0f} MB, {stats['n_docs']:,} unique docs")
    lsh.clear()
    gc.collect()


# ── stage 4: train tokenizer ────────────────────────────────────────
def stage_train_tokenizer():
    banner("Stage 4: Train tokenizer")

    model_path = Path(f"{TOKENIZER_NEW}.model")
    if model_path.exists():
        print(f"[skip] Tokenizer exists: {model_path}")
        return

    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_tokenizer.py"),
        "--raw-dir", str(DEDUPED_DIR),
        "--output", str(TOKENIZER_NEW),
        "--vocab-size", "48000",
        "--sample-mb", "2000",
    ]
    if TOKENIZER_OLD.exists():
        cmd += ["--compare-old", str(TOKENIZER_OLD)]

    print("Training tokenizer (48k vocab, 2 GB sample)...")
    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print("WARNING: Tokenizer training failed, falling back to v4.")
        if TOKENIZER_OLD.exists():
            shutil.copy2(str(TOKENIZER_OLD), str(model_path))


# ── stage 5: mix and shard ──────────────────────────────────────────
def stage_mix_and_shard():
    banner("Stage 5: Domain mix + tokenization")

    meta_path = PRETRAIN_SHARDS / "meta.yaml"
    if meta_path.exists():
        print(f"[skip] Pretrain shards exist: {PRETRAIN_SHARDS}")
        return

    import sentencepiece as spm
    from src.data.mixer import DataSource, DataMixer

    tok_path = Path(f"{TOKENIZER_NEW}.model")
    if not tok_path.exists():
        tok_path = TOKENIZER_OLD
    print(f"Tokenizer: {tok_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(tok_path))
    eos_id = sp.eos_id()
    vocab_size = sp.get_piece_size()

    sources = []
    source_configs = [
        ("fineweb_edu", "fineweb_edu_hq.txt", 0.45),
        ("wikipedia", "wikipedia_clean.txt", 0.15),
        ("stackexchange", "stackexchange_clean.txt", 0.10),
        ("code", "code_clean.txt", 0.15),
        ("arxiv", "arxiv_clean.txt", 0.05),
    ]

    v4_raw = ROOT / "data" / "raw_v4"
    v4_configs = [
        ("v4_pretrain", "pretrain_corpus.txt", 0.05),
        ("v4_fineweb", "fineweb_edu_corpus.txt", 0.05),
    ]

    for name, filename, weight in source_configs:
        path = DEDUPED_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            sources.append(DataSource(name, path, weight))

    for name, filename, weight in v4_configs:
        path = v4_raw / filename
        if path.exists() and path.stat().st_size > 0:
            sources.append(DataSource(name, path, weight))

    if not sources:
        print("ERROR: No data sources found!")
        sys.exit(1)

    mixer = DataMixer(sources)
    print(mixer.summary())

    pack_len = SEQ_LEN + 1
    PRETRAIN_SHARDS.mkdir(parents=True, exist_ok=True)

    print("\nTokenizing...")
    t0 = time.time()
    token_chunks = []
    total_tokens = 0
    source_counts = {}
    doc_count = 0

    for source_name, doc_text in mixer.mix():
        tokens = sp.encode(doc_text, out_type=int)
        tokens.append(eos_id)
        token_chunks.append(np.array(tokens, dtype=np.uint16))
        n_toks = len(tokens)
        total_tokens += n_toks
        source_counts[source_name] = source_counts.get(source_name, 0) + n_toks
        doc_count += 1

        if doc_count % 100000 == 0:
            print(f"  {doc_count:,} docs | {total_tokens / 1e9:.3f}B tokens")

        if len(token_chunks) > 500000:
            token_chunks = [np.concatenate(token_chunks)]
            gc.collect()

    dt = time.time() - t0
    print(f"\n  Total: {total_tokens:,} tokens ({total_tokens / 1e9:.3f}B) in {elapsed_str(dt)}")
    print(f"\n  By source:")
    for name, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / total_tokens * 100
        print(f"    {name:20s} {count / 1e9:.3f}B ({pct:.1f}%)")

    print("  Packing...")
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
        path = PRETRAIN_SHARDS / f"train_{shard_idx:04d}.bin"
        packed[start:end].tofile(str(path))
        if shard_idx % 10 == 0:
            print(f"  train_{shard_idx:04d}: {end - start:,} seqs")
        shard_idx += 1

    print(f"  {shard_idx} train shards")

    val_path = PRETRAIN_SHARDS / "val_0000.bin"
    packed[n_train:].tofile(str(val_path))
    print(f"  val_0000: {n_val:,} seqs")

    meta = {
        "total_tokens": int(total_tokens),
        "seq_len": SEQ_LEN,
        "pack_len": pack_len,
        "n_train_sequences": int(n_train),
        "n_val_sequences": int(n_val),
        "n_train_shards": shard_idx,
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "tokenizer": str(tok_path.name),
        "source_tokens": {k: int(v) for k, v in source_counts.items()},
    }
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)

    del packed
    gc.collect()
    print(f"\nPretrain shards: {n_train:,} train + {n_val:,} val")


# ── stage 6: pretrain ────────────────────────────────────────────────
def stage_pretrain():
    banner("Stage 6: Pretrain (500M, 200k steps)")

    pretrain_ckpt = CKPT_DIR / "pretrain_1.1_final.pt"
    if pretrain_ckpt.exists():
        print(f"[skip] Pretrain checkpoint exists: {pretrain_ckpt}")
        return

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(CKPT_DIR.glob("1.1_step_*.pt"))
    if ckpts:
        resume_from = ckpts[-1]
        print(f"Resuming from: {resume_from}")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(PRETRAIN_CFG),
               "--resume", str(resume_from)]
    else:
        print("Starting pretraining from scratch")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(PRETRAIN_CFG)]

    print(f"  Config: {PRETRAIN_CFG}")
    print(f"  Stop and resume anytime.\n")

    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print(f"WARNING: Training exited with code {result.returncode}")

    final = CKPT_DIR / "final.pt"
    if final.exists():
        shutil.copy2(str(final), str(pretrain_ckpt))
        print(f"\nPretrain checkpoint: {pretrain_ckpt}")
    else:
        ckpts = sorted(CKPT_DIR.glob("1.1_step_*.pt"))
        if ckpts:
            shutil.copy2(str(ckpts[-1]), str(pretrain_ckpt))
            print(f"\nUsing latest: {ckpts[-1]} -> {pretrain_ckpt}")
        else:
            print("ERROR: No checkpoint found!")
            sys.exit(1)


# ── stage 7: synthetic instruct ─────────────────────────────────────
def stage_generate_synthetic():
    banner("Stage 7: Synthetic instruction data")

    synthetic_path = RAW_DIR / "synthetic_instruct.jsonl"

    if synthetic_path.exists() and synthetic_path.stat().st_size > 50_000_000:
        n_lines = sum(1 for _ in open(synthetic_path))
        print(f"[skip] Synthetic data: {n_lines:,} samples ({synthetic_path.stat().st_size / 1e6:.1f} MB)")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Skipping synthetic generation.")
        print("  To generate:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    python scripts/generate_synthetic.py --n-samples 50000")
        print("\n  Optional but highly recommended.")
        return

    cmd = [
        sys.executable, str(ROOT / "scripts" / "generate_synthetic.py"),
        "--n-samples", "50000",
        "--output", str(synthetic_path),
        "--model", "claude-haiku-4-5-20251001",
    ]
    if synthetic_path.exists():
        cmd.append("--resume")

    print("Generating 50k synthetic samples (~$15-20 with Haiku)...")
    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print(f"WARNING: Generation exited with code {result.returncode}")


# ── stage 8: build instruct shards ──────────────────────────────────
def stage_build_instruct_shards():
    banner("Stage 8: Build instruct shards")

    meta_path = INSTRUCT_SHARDS / "meta.yaml"
    if meta_path.exists():
        print(f"[skip] Instruct shards exist: {INSTRUCT_SHARDS}")
        return

    import sentencepiece as spm

    tok_path = Path(f"{TOKENIZER_NEW}.model")
    if not tok_path.exists():
        tok_path = TOKENIZER_OLD

    sp = spm.SentencePieceProcessor()
    sp.load(str(tok_path))
    eos_id = sp.eos_id()

    pack_len = SEQ_LEN + 1
    INSTRUCT_SHARDS.mkdir(parents=True, exist_ok=True)

    instruct_files = []
    synthetic = RAW_DIR / "synthetic_instruct.jsonl"
    if synthetic.exists():
        instruct_files.append(("synthetic", synthetic))

    v4_instruct = ROOT / "data" / "raw_v4" / "instruct_corpus.jsonl"
    if v4_instruct.exists():
        instruct_files.append(("v4_instruct", v4_instruct))

    if not instruct_files:
        print("ERROR: No instruct data found!")
        sys.exit(1)

    print(f"Sources:")
    for name, path in instruct_files:
        print(f"  {name}: {path.stat().st_size / 1e6:.1f} MB")

    print("\nTokenizing with loss masking...")
    t0 = time.time()

    all_tokens = []
    all_masks = []
    total_tokens = 0
    n_convs = 0
    n_multiturn = 0
    source_counts = {}

    for source_name, path in instruct_files:
        file_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data["text"]
                except (json.JSONDecodeError, KeyError):
                    continue

                tokens, mask = _build_multiturn_mask(text, sp)
                if tokens is None:
                    continue

                if text.count("User:") > 1:
                    n_multiturn += 1

                tokens.append(eos_id)
                mask = np.append(mask, 1)

                all_tokens.extend(tokens)
                all_masks.extend(mask)
                total_tokens += len(tokens)
                n_convs += 1
                file_count += 1

                if n_convs % 100000 == 0:
                    print(f"  {n_convs:,} convs | {total_tokens / 1e6:.1f}M tokens")

        source_counts[source_name] = file_count
        print(f"  {source_name}: {file_count:,} conversations")

    dt = time.time() - t0
    print(f"\n  Total: {n_convs:,} convs, {total_tokens / 1e6:.1f}M tokens ({elapsed_str(dt)})")
    print(f"  Multi-turn: {n_multiturn:,} ({n_multiturn / max(1, n_convs) * 100:.1f}%)")

    print("  Packing...")
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
        (INSTRUCT_SHARDS / f"train_{shard_idx:04d}.bin").write_bytes(
            packed_tokens[start:end].tobytes())
        (INSTRUCT_SHARDS / f"train_mask_{shard_idx:04d}.bin").write_bytes(
            packed_masks[start:end].tobytes())
        print(f"  train_{shard_idx:04d}: {end - start:,} seqs")
        shard_idx += 1

    (INSTRUCT_SHARDS / "val_0000.bin").write_bytes(packed_tokens[n_train:].tobytes())
    (INSTRUCT_SHARDS / "val_mask_0000.bin").write_bytes(packed_masks[n_train:].tobytes())
    print(f"  val_0000: {n_val:,} seqs")

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
        "source_counts": source_counts,
    }
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)

    total_masked = int(packed_masks[:n_train].sum())
    pct = total_masked / (n_train * pack_len) * 100
    print(f"\nInstruct shards: {n_train:,} train + {n_val:,} val")
    print(f"  Loss mask: {pct:.1f}% assistant tokens")

    del packed_tokens, packed_masks
    gc.collect()


def _build_multiturn_mask(text, sp):
    """Build loss mask for multi-turn conversation."""
    tokens = sp.encode(text, out_type=int)
    if len(tokens) < 4:
        return None, None

    mask = np.zeros(len(tokens), dtype=np.uint8)
    pos = 0
    while pos < len(text):
        asst_start = text.find("Assistant:", pos)
        if asst_start < 0:
            break
        next_user = text.find("\nUser:", asst_start + 10)
        asst_end = next_user if next_user >= 0 else len(text)

        prefix_tokens = sp.encode(text[:asst_start], out_type=int)
        segment_tokens = sp.encode(text[:asst_end], out_type=int)
        mask[len(prefix_tokens):len(segment_tokens)] = 1
        pos = asst_end + 1

    return tokens, mask


# ── stage 9: finetune ────────────────────────────────────────────────
def stage_finetune():
    banner("Stage 9: Finetune (30k steps)")

    finetune_ckpt = CKPT_DIR / "finetune_1.1_final.pt"
    if finetune_ckpt.exists():
        print(f"[skip] Finetune checkpoint exists: {finetune_ckpt}")
        return

    pretrain_ckpt = CKPT_DIR / "pretrain_1.1_final.pt"
    if not pretrain_ckpt.exists():
        print("ERROR: No pretrain checkpoint!")
        sys.exit(1)

    ft_ckpts = sorted(CKPT_DIR.glob("1.1_ft_step_*.pt"))
    if ft_ckpts:
        resume = ft_ckpts[-1]
        print(f"Resuming finetune from: {resume}")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(FINETUNE_CFG),
               "--resume", str(resume)]
    else:
        print(f"Starting finetune from: {pretrain_ckpt}")
        cmd = [sys.executable, "-m", "src.train.train",
               "--config", str(FINETUNE_CFG),
               "--finetune", str(pretrain_ckpt)]

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"WARNING: Finetune exited with code {result.returncode}")

    final = CKPT_DIR / "final.pt"
    if final.exists():
        shutil.copy2(str(final), str(finetune_ckpt))
        print(f"\nFinetune checkpoint: {finetune_ckpt}")
    else:
        ft_ckpts = sorted(CKPT_DIR.glob("1.1_ft_step_*.pt"))
        if ft_ckpts:
            shutil.copy2(str(ft_ckpts[-1]), str(finetune_ckpt))


# ── stage 10: test ───────────────────────────────────────────────────
def stage_test():
    banner("Stage 10: Test")

    for ckpt_name in ["finetune_1.1_final.pt", "pretrain_1.1_final.pt"]:
        ckpt = CKPT_DIR / ckpt_name
        if ckpt.exists():
            break
    else:
        print("ERROR: No checkpoint found!")
        return

    print(f"Testing: {ckpt}")

    test_prompts = [
        "What color is the sky?",
        "What is 2 + 2?",
        "Who was the first president of the United States?",
        "Explain how photosynthesis works in 3 sentences.",
        "Write a Python function that checks if a number is prime.",
        "What is the difference between weather and climate?",
        "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "Write a short poem about the ocean.",
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

        tok_path = Path(f"{TOKENIZER_NEW}.model")
        if not tok_path.exists():
            tok_path = TOKENIZER_OLD

        sp = spm.SentencePieceProcessor()
        sp.load(str(tok_path))

        print(f"Model: {model.count_parameters():,} params")
        print(f"Tokenizer: {tok_path.name} ({sp.get_piece_size():,} vocab)\n")

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
            print(f"A: {response.strip()}\n")

    except Exception as e:
        print(f"Test failed: {e}")
        print(f"\nManual test:")
        print(f"  python -m src.inference.chat --checkpoint {ckpt} --config {FINETUNE_CFG}")


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plasma 1.1 training pipeline")
    parser.add_argument("--stage", choices=[
        "cleanup", "download", "quality", "dedup", "tokenizer",
        "shards", "pretrain", "synthetic", "instruct", "finetune", "test",
    ], help="Run a specific stage")
    args = parser.parse_args()

    print("=" * 64)
    print("  Plasma 1.1 training pipeline")
    print("=" * 64)

    stages = {
        "cleanup":   stage_cleanup,
        "download":  stage_download,
        "quality":   stage_quality_score,
        "dedup":     stage_minhash_dedup,
        "tokenizer": stage_train_tokenizer,
        "shards":    stage_mix_and_shard,
        "pretrain":  stage_pretrain,
        "synthetic": stage_generate_synthetic,
        "instruct":  stage_build_instruct_shards,
        "finetune":  stage_finetune,
        "test":      stage_test,
    }

    if args.stage:
        stages[args.stage]()
    else:
        for name, fn in stages.items():
            fn()

    print("\n" + "=" * 64)
    print("  Done.")
    print(f"    python -m src.inference.chat --checkpoint checkpoints/finetune_1.1_final.pt --config {FINETUNE_CFG}")
    print("    python run.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
