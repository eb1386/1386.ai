#!/usr/bin/env python3
# 1.1 training pipeline

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# windows console encoding fix
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# paths
RAW_DIR          = ROOT / "data" / "raw_1.1"
SCORED_DIR       = ROOT / "data" / "scored_1.1"
DEDUPED_DIR      = ROOT / "data" / "deduped_1.1"
PRETRAIN_SHARDS  = ROOT / "data" / "shards_1.1"
INSTRUCT_SHARDS  = ROOT / "data" / "instruct_shards_1.1"
TOKENIZER_OLD    = ROOT / "data" / "tokenizer_1.0.model"
TOKENIZER_NEW    = ROOT / "data" / "tokenizer_1.1"
QUALITY_MODEL    = ROOT / "data" / "quality_classifier.pkl"
TOXICITY_MODEL   = ROOT / "data" / "toxicity_classifier.pkl"
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


# ── stage -1: verify + repair ───────────────────────────────────────
def _ckpt_step(p):
    import re
    m = re.search(r"_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else 0


def stage_verify():
    """Detect corruption from the pre-fix finetune-resume bug and repair.

    Symptom: an older bug in train.py wrote finetune weights to
    pretrain_1.1_final.pt when --resume (not --finetune) was used.
    Detect by reading the saved 'step' from inside the checkpoint
    against the pretrain config's max_steps; if low, restore the
    pretrain final from the highest-step pretrain checkpoint.
    Also wipe finetune artifacts that were trained with bad masks.
    """
    banner("Stage -1: Verify pretrain integrity + reset finetune state")

    pretrain_ckpt = CKPT_DIR / "pretrain_1.1_final.pt"
    pretrain_steps = []
    for p in CKPT_DIR.glob("1.1_step_*.pt"):
        pretrain_steps.append(p)
    pretrain_steps.sort(key=_ckpt_step)

    needs_restore = False
    if pretrain_ckpt.exists():
        try:
            import torch
            ckpt = torch.load(str(pretrain_ckpt), map_location="cpu", weights_only=False)
            saved_step = int(ckpt.get("step", 0))
            saved_prefix = (
                ckpt.get("config", {}).get("training", {}).get("checkpoint_prefix", "")
            )
            print(f"  pretrain_1.1_final.pt: step={saved_step:,} prefix={saved_prefix!r}")

            if "_ft" in saved_prefix:
                print("  WARNING: pretrain_1.1_final.pt was written by a finetune run "
                      "(checkpoint_prefix contains '_ft'). It is corrupted.")
                needs_restore = True
            elif saved_step < 100_000:
                print(f"  WARNING: pretrain step={saved_step} looks too low — "
                      "likely overwritten by old finetune-resume bug.")
                needs_restore = True
            else:
                print("  pretrain checkpoint looks healthy")
        except Exception as e:
            print(f"  ERROR reading pretrain checkpoint: {e}")
            needs_restore = True
    else:
        print("  pretrain_1.1_final.pt missing — will need to be created")
        needs_restore = True

    if needs_restore:
        if pretrain_steps:
            best = pretrain_steps[-1]
            print(f"  Restoring pretrain_1.1_final.pt from {best.name}")
            shutil.copy2(str(best), str(pretrain_ckpt))
        else:
            print("  No 1.1_step_*.pt to restore from. Pretrain stage will start fresh.")
            if pretrain_ckpt.exists():
                pretrain_ckpt.unlink()
                print("  Removed corrupted pretrain_1.1_final.pt")

    # wipe finetune artifacts (masks were broken; data was insufficient).
    # also wipe instruct shards built with the broken mask builder.
    finetune_ckpt = CKPT_DIR / "finetune_1.1_final.pt"
    if finetune_ckpt.exists():
        print(f"  Removing {finetune_ckpt.name} (will be rebuilt with correct masks)")
        finetune_ckpt.unlink()

    ft_steps = list(CKPT_DIR.glob("1.1_ft_step_*.pt"))
    for p in ft_steps:
        print(f"  Removing {p.name}")
        p.unlink()

    if INSTRUCT_SHARDS.exists() and any(INSTRUCT_SHARDS.iterdir()):
        print(f"  Removing instruct_shards_1.1/ (built with broken mask builder)")
        shutil.rmtree(str(INSTRUCT_SHARDS))


# ── stage 0: cleanup ────────────────────────────────────────────────
def stage_cleanup():
    banner("Stage 0: Cleanup")

    freed = 0
    if CKPT_DIR.exists():
        for pattern in ["1.1_step_*.pt", "1.1_ft_step_*.pt"]:
            import re
            def _step_num(p):
                m = re.search(r'_(\d+)\.pt$', p.name)
                return int(m.group(1)) if m else 0
            ckpts = sorted(CKPT_DIR.glob(pattern), key=_step_num)
            if len(ckpts) > 1:
                # keep latest, delete rest
                for ckpt in ckpts[:-1]:
                    freed += ckpt.stat().st_size
                    ckpt.unlink()
                print(f"  Cleaned {len(ckpts) - 1} old checkpoints, kept {ckpts[-1].name}")
            elif ckpts:
                print(f"  [kept] {ckpts[0].name} (resume)")

    if freed > 0:
        print(f"  Freed {freed / 1e9:.1f} GB")
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

    print(f"\n{'-' * 40}")
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
    if path.exists() and path.stat().st_size >= FINEWEB_TARGET * 0.5:
        print(f"[skip] FineWeb-Edu HQ ({path.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"Downloading FineWeb-Edu (score >= 3, target: {FINEWEB_TARGET / 1e9:.0f} GB)...")
    total_chars = 0
    doc_count = 0
    skipped = 0
    t0 = time.time()

    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True, trust_remote_code=False)

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
                          split="train", streaming=True, trust_remote_code=False)

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
        ds = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                          split="train", streaming=True, trust_remote_code=False)

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
    languages = ["python", "javascript"]

    try:
        with open(path, "w", encoding="utf-8") as f:
            per_lang = CODE_TARGET // len(languages)
            for lang in languages:
                lang_chars = 0
                lang_docs = 0
                print(f"\n  Downloading {lang}...")

                try:
                    try:
                        ds = load_dataset(
                            "bigcode/starcoderdata",
                            data_dir=lang,
                            split="train",
                            streaming=True,
                            trust_remote_code=False,
                        )
                        field = "content"
                    except Exception as e:
                        print(f"    starcoderdata unavailable for {lang}: {e}")
                        if lang == "python":
                            print("    Falling back to codeparrot/codeparrot-clean...")
                            ds = load_dataset(
                                "codeparrot/codeparrot-clean",
                                split="train",
                                streaming=True,
                            )
                            field = "content"
                        else:
                            print(f"    Falling back to code-search-net/code_search_net {lang}...")
                            ds = load_dataset(
                                "code-search-net/code_search_net",
                                name=lang,
                                split="train",
                                streaming=True,
                            )
                            field = "whole_func_string"
                    for example in ds:
                        content = example.get(field, "").strip()
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
                          split="train", streaming=True, trust_remote_code=False)
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


# ── stage 2: train classifiers ──────────────────────────────────────
def stage_train_classifiers():
    banner("Stage 2: Train quality + toxicity classifiers")

    from src.data.classifier import train_quality_classifier, train_toxicity_classifier

    if QUALITY_MODEL.exists():
        print(f"[skip] Quality classifier exists: {QUALITY_MODEL}")
    else:
        try:
            train_quality_classifier(str(QUALITY_MODEL), n_samples=200_000)
        except Exception as e:
            print(f"  Quality classifier failed: {e}")
            print("  Falling back to heuristic scoring only.")

    if TOXICITY_MODEL.exists():
        print(f"[skip] Toxicity classifier exists: {TOXICITY_MODEL}")
    else:
        try:
            train_toxicity_classifier(str(TOXICITY_MODEL), n_samples=200_000)
        except Exception as e:
            print(f"  Toxicity classifier failed: {e}")
            print("  Skipping toxicity filtering.")


# ── stage 3: quality + toxicity scoring ─────────────────────────────
def stage_quality_score():
    banner("Stage 3: Quality + toxicity scoring")

    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    from src.data.quality import filter_and_score as heuristic_score

    source_files = [
        ("fineweb_edu_hq.txt", QUALITY_MIN_SCORE),
        ("wikipedia_clean.txt", 0.45),
        ("stackexchange_clean.txt", QUALITY_MIN_SCORE),
        ("code_clean.txt", 0.40),
        ("arxiv_clean.txt", 0.45),
    ]

    # short-circuit if every source is already scored or unavailable — avoids
    # loading the classifier pickles (which can fail on numpy version drift)
    pending = []
    for f, t in source_files:
        dst = SCORED_DIR / f
        src = RAW_DIR / f
        if dst.exists() and dst.stat().st_size > 0:
            continue  # already scored
        if not src.exists() or src.stat().st_size == 0:
            continue  # raw data missing or empty
        pending.append((f, t))
    if not pending:
        for filename, _ in source_files:
            if (SCORED_DIR / filename).exists():
                print(f"[skip] {filename} already scored")
            elif not (RAW_DIR / filename).exists():
                print(f"[skip] {filename} not downloaded")
        return

    # load trained classifiers if available, only when there's work to do
    quality_clf = None
    toxicity_clf = None

    if QUALITY_MODEL.exists():
        try:
            from src.data.classifier import QualityClassifier
            quality_clf = QualityClassifier(str(QUALITY_MODEL))
            print("  Using trained quality classifier")
        except Exception as e:
            print(f"  Quality classifier failed to load ({e.__class__.__name__}); "
                  "falling back to heuristics")
    else:
        print("  No quality classifier found, using heuristics only")

    if TOXICITY_MODEL.exists():
        try:
            from src.data.classifier import ToxicityClassifier
            toxicity_clf = ToxicityClassifier(str(TOXICITY_MODEL))
            print("  Using trained toxicity classifier")
        except Exception as e:
            print(f"  Toxicity classifier failed to load ({e.__class__.__name__}); "
                  "skipping toxicity filter")
    else:
        print("  No toxicity classifier found, skipping toxicity filter")

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
        toxic_dropped = 0
        quality_dropped = 0
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

                    # toxicity filter first (fast reject)
                    if toxicity_clf and toxicity_clf.is_toxic(text, threshold=0.5):
                        toxic_dropped += 1
                        continue

                    # combined quality: classifier + heuristics
                    if quality_clf:
                        clf_score = quality_clf.score(text)
                        _, heur_score = heuristic_score(text, 0.0)
                        # weighted blend: 60% classifier, 40% heuristics
                        combined = 0.6 * clf_score + 0.4 * heur_score
                        passed = combined >= threshold
                    else:
                        passed, _ = heuristic_score(text, threshold)

                    if passed:
                        fout.write(text + "\n\n")
                        kept += 1
                    else:
                        quality_dropped += 1

                    if total % 100000 == 0:
                        pct = kept / total * 100
                        print(f"  {total:,} scored | {kept:,} kept ({pct:.1f}%) | "
                              f"toxic: {toxic_dropped:,} | low-quality: {quality_dropped:,}")

            if doc_buffer:
                text = "\n".join(doc_buffer).strip()
                if len(text) >= 50:
                    total += 1
                    if toxicity_clf and toxicity_clf.is_toxic(text, threshold=0.5):
                        toxic_dropped += 1
                    else:
                        if quality_clf:
                            clf_score = quality_clf.score(text)
                            _, heur_score = heuristic_score(text, 0.0)
                            combined = 0.6 * clf_score + 0.4 * heur_score
                            passed = combined >= threshold
                        else:
                            passed, _ = heuristic_score(text, threshold)
                        if passed:
                            fout.write(text + "\n\n")
                            kept += 1
                        else:
                            quality_dropped += 1

        dt = time.time() - t0
        pct = kept / max(1, total) * 100
        print(f"  {filename}: {kept:,}/{total:,} kept ({pct:.1f}%) | "
              f"toxic: {toxic_dropped:,} | low-quality: {quality_dropped:,} | {elapsed_str(dt)}")


# ── stage 3: minhash dedup ──────────────────────────────────────────
def stage_minhash_dedup():
    banner("Stage 4: MinHash dedup")

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
    banner("Stage 5: Train tokenizer")

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
        print("WARNING: Tokenizer training failed, falling back to 1.0.")
        if TOKENIZER_OLD.exists():
            shutil.copy2(str(TOKENIZER_OLD), str(model_path))


# ── stage 5: mix and shard ──────────────────────────────────────────
def stage_mix_and_shard():
    banner("Stage 6: Domain mix + tokenization")

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

    old_raw = ROOT / "data" / "raw_1.0"
    old_configs = [
        ("1.0_pretrain", "pretrain_corpus.txt", 0.05),
        ("1.0_fineweb", "fineweb_edu_corpus.txt", 0.05),
    ]

    for name, filename, weight in source_configs:
        path = DEDUPED_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            sources.append(DataSource(name, path, weight))

    for name, filename, weight in old_configs:
        path = old_raw / filename
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
    banner("Stage 7: Pretrain (500M, 200k steps)")

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

    # train.py writes pretrain_1.1_final.pt directly when training completes;
    # if the run was interrupted, fall back to the latest step checkpoint.
    if pretrain_ckpt.exists():
        print(f"\nPretrain checkpoint: {pretrain_ckpt}")
    else:
        step_ckpts = sorted(CKPT_DIR.glob("1.1_step_*.pt"), key=_ckpt_step)
        if step_ckpts:
            shutil.copy2(str(step_ckpts[-1]), str(pretrain_ckpt))
            print(f"\nTraining did not complete; using latest step checkpoint: "
                  f"{step_ckpts[-1].name} -> {pretrain_ckpt.name}")
        else:
            print("ERROR: No checkpoint found!")
            sys.exit(1)


# ── stage 7a: download public instruct datasets ────────────────────
def stage_download_instruct():
    banner("Stage 8a: Download free public instruct datasets")

    have = sorted(RAW_DIR.glob("instruct_*.jsonl"))
    total = 0
    for p in have:
        if p.stat().st_size > 0:
            total += sum(1 for _ in open(p, encoding="utf-8"))
    # threshold tuned to require all 10 sources downloaded; the per-source
    # downloader has its own skip-when-file-exists logic, so re-running is cheap
    if total >= 700_000:
        print(f"[skip] Already have {total:,} public instruct samples")
        return

    cmd = [sys.executable, str(ROOT / "scripts" / "download_instruct.py")]
    print("Downloading public instruction datasets (no API key needed)...")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"WARNING: download_instruct exited with code {result.returncode}")


# ── stage 7b: synthetic instruct ────────────────────────────────────
def stage_generate_synthetic():
    banner("Stage 8b: Synthetic instruction data (optional, voice/style)")

    synthetic_path = RAW_DIR / "synthetic_instruct.jsonl"

    if synthetic_path.exists():
        n_lines = sum(1 for _ in open(synthetic_path, encoding="utf-8"))
        if n_lines >= 5000:
            print(f"[skip] Synthetic data: {n_lines:,} samples ({synthetic_path.stat().st_size / 1e6:.1f} MB)")
            return
        print(f"  Found {n_lines:,} existing samples, need more...")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        if synthetic_path.exists():
            print("No API key but synthetic data exists. Continuing with what we have.")
            return
        print("ANTHROPIC_API_KEY not set. Skipping synthetic generation.")
        print("  To generate:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    python scripts/generate_synthetic.py --n-samples 15000")
        print("\n  Optional but highly recommended.")
        return

    cmd = [
        sys.executable, str(ROOT / "scripts" / "generate_synthetic.py"),
        "--n-samples", "15000",
        "--output", str(synthetic_path),
        "--model", "claude-haiku-4-5-20251001",
        "--max-tokens", "512",
    ]
    if synthetic_path.exists():
        cmd.append("--resume")

    print("Generating synthetic samples...")
    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        print(f"WARNING: Generation exited with code {result.returncode}")


# ── stage 8: build instruct shards ──────────────────────────────────
def stage_build_instruct_shards():
    banner("Stage 9: Build instruct shards")

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

    # 1.0's FLAN-style instruct corpus (592k samples) — the proven recipe that
    # gave 1.0 its instruction-following quality. include it as the foundation
    # for 1.1 SFT, since 1.1 is "1.0 scaled up" architecturally.
    flan_1_0 = ROOT / "data" / "raw_1.0" / "instruct_corpus.jsonl"
    if flan_1_0.exists() and flan_1_0.stat().st_size > 0:
        instruct_files.append(("flan_1.0", flan_1_0))

    # public instruct datasets pulled by scripts/download_instruct.py
    # each is included once. their volume already gives plenty of coverage.
    public_files = sorted(RAW_DIR.glob("instruct_*.jsonl"))
    for p in public_files:
        if p.stat().st_size > 0:
            instruct_files.append((p.stem, p))

    # synthetic data: repeat for several epochs since it's small but high quality.
    # 4 epochs is the sweet spot — 8 caused overfitting to the small unique set.
    synthetic = RAW_DIR / "synthetic_instruct.jsonl"
    if synthetic.exists():
        public_total = sum(
            1 for p in public_files for _ in open(p, encoding="utf-8")
        ) if public_files else 0
        epochs = 4 if public_total >= 50_000 else 10
        for i in range(epochs):
            instruct_files.append((f"synthetic_ep{i}", synthetic))

    if not instruct_files:
        print("ERROR: No instruct data found!")
        print("  Run: python scripts/download_instruct.py")
        print("  Or:  python scripts/generate_synthetic.py --n-samples 15000")
        sys.exit(1)

    print(f"Sources:")
    for name, path in instruct_files:
        print(f"  {name}: {path.stat().st_size / 1e6:.1f} MB")

    print("\nTokenizing with loss masking...")
    t0 = time.time()

    # collect each conversation as its own (tokens, mask) pair; we'll shuffle
    # at conversation granularity before packing so same-source clusters
    # don't end up packed together inside 1024-token sequences.
    conv_tokens = []  # list of np.uint16 arrays
    conv_masks = []   # list of np.uint8 arrays
    total_tokens = 0
    n_convs = 0
    n_multiturn = 0
    source_counts = {}
    skipped_counts = {}

    for source_name, path in instruct_files:
        file_count = 0
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data["text"]
                except (json.JSONDecodeError, KeyError):
                    skipped += 1
                    continue

                text = _normalize_instruct_text(text)
                if _has_bad_sft_pattern(text):
                    skipped += 1
                    continue

                tokens, mask = _build_multiturn_mask(text, sp)
                if tokens is None:
                    skipped += 1
                    continue

                if text.count("User:") > 1:
                    n_multiturn += 1

                tokens.append(eos_id)
                mask = np.append(mask, 1)

                conv_tokens.append(np.array(tokens, dtype=np.uint16))
                conv_masks.append(np.asarray(mask, dtype=np.uint8))
                total_tokens += len(tokens)
                n_convs += 1
                file_count += 1

                if n_convs % 100000 == 0:
                    print(f"  {n_convs:,} convs | {total_tokens / 1e6:.1f}M tokens")

        source_counts[source_name] = file_count
        skipped_counts[source_name] = skipped
        print(f"  {source_name}: {file_count:,} conversations"
              f" ({skipped:,} skipped)")

    dt = time.time() - t0
    print(f"\n  Total: {n_convs:,} convs, {total_tokens / 1e6:.1f}M tokens ({elapsed_str(dt)})")
    print(f"  Multi-turn: {n_multiturn:,} ({n_multiturn / max(1, n_convs) * 100:.1f}%)")

    # shuffle conversations BEFORE packing to mix sources within sequences
    print("  Shuffling conversations...")
    perm_rng = np.random.default_rng(42)
    perm = perm_rng.permutation(n_convs)
    conv_tokens = [conv_tokens[i] for i in perm]
    conv_masks = [conv_masks[i] for i in perm]

    print("  Packing...")
    all_tokens = np.concatenate(conv_tokens)
    all_masks = np.concatenate(conv_masks)
    del conv_tokens, conv_masks
    gc.collect()

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
        "skipped_counts": skipped_counts,
    }
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)

    total_masked = int(packed_masks[:n_train].sum())
    pct = total_masked / (n_train * pack_len) * 100
    print(f"\nInstruct shards: {n_train:,} train + {n_val:,} val")
    print(f"  Loss mask: {pct:.1f}% assistant tokens")

    del packed_tokens, packed_masks
    gc.collect()


_ROLE_RE = re.compile(r"(?:^|\n)(User|Assistant): ?")
_MARKDOWN_ROLE_RE = re.compile(
    r"(?m)^\s*(?:[-*]\s*)?(?:#{1,6}\s*)?(?:\*\*)?"
    r"(User|Assistant)(?:\s*(?:turn|response|message))?"
    r"(?:\s*\d+)?\s*:(?:\*\*)?\s*"
)
_TURN_HEADING_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?:\*\*)?"
    r"Turn\s+\d+(?:\s*[-:][^\n]*)?(?:\*\*)?\s*$"
)
_BAD_SFT_PATTERNS = (
    "i am a language model",
    "i am an ai",
    "i'm an ai",
    "i am a computer program",
    "as an ai",
    "as a language model",
    "i cannot answer",
    "i can't answer",
    "i do not have access",
    "i don't have access",
    "i am not able to provide",
    "i am unable to",
    "as an artificial intelligence",
    "i'm just an ai",
    "i'm a language model",
    "i don't have personal",
    "i do not have personal",
    "as a helpful assistant",
    "i am only a",
    "i'm only a",
)


def _normalize_instruct_text(text: str) -> str:
    """Normalize common generated conversation wrappers into User/Assistant."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = _TURN_HEADING_RE.sub("", text)
    text = _MARKDOWN_ROLE_RE.sub(lambda m: f"{m.group(1)}: ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_bad_sft_pattern(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _BAD_SFT_PATTERNS)


def _build_multiturn_mask(text, sp):
    """Build loss mask by tokenizing each role segment separately.

    Two key behaviors:
    1. Role boundaries are line-anchored (^ or after \\n) so role markers
       inside content (code, JSON, quoted dialogs) don't split responses.
    2. The role label itself ("Assistant: " / "User: ") is NEVER masked.
       The label is part of the prompt format the user provides, not
       something the model should learn to generate. If we mask=1 on the
       label, the model learns to emit "Assistant:" mid-response, which
       is exactly what we don't want.

    For each segment: tokenize the label (mask=0) and the content
    (mask=1 for assistant, 0 for user) separately.
    """
    if not text or not text.strip():
        return None, None

    # locate role boundaries with a line-anchored regex.
    # m.start(1) = where "User"/"Assistant" begins
    # m.end()    = where the colon (and optional space) ends — i.e. where
    #              the actual content starts
    boundaries = []
    for m in _ROLE_RE.finditer(text):
        role_start = m.start(1)
        label_end = m.end()
        role = "user" if m.group(1) == "User" else "assistant"
        boundaries.append((role, role_start, label_end))

    if not boundaries:
        return None, None

    tokens = []
    mask_bits = []

    # any text before the first role marker (rare) is treated as non-assistant
    if boundaries[0][1] > 0:
        prefix = text[: boundaries[0][1]]
        if prefix.strip():
            ptoks = sp.encode(prefix, out_type=int)
            tokens.extend(ptoks)
            mask_bits.extend([0] * len(ptoks))

    for i, (role, role_start, label_end) in enumerate(boundaries):
        next_start = (
            boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
        )

        # 1. role label "User: " or "Assistant: " — always mask=0
        label_text = text[role_start:label_end]
        if label_text:
            label_tokens = sp.encode(label_text, out_type=int)
            tokens.extend(label_tokens)
            mask_bits.extend([0] * len(label_tokens))

        # 2. content after the label, up to the next role marker
        content_text = text[label_end:next_start]
        if content_text:
            content_tokens = sp.encode(content_text, out_type=int)
            bit = 1 if role == "assistant" else 0
            tokens.extend(content_tokens)
            mask_bits.extend([bit] * len(content_tokens))

    if len(tokens) < 4:
        return None, None
    if not any(mask_bits):
        # no assistant content at all
        return None, None

    return tokens, np.array(mask_bits, dtype=np.uint8)


# ── stage 9: finetune ────────────────────────────────────────────────
def stage_finetune():
    # max_steps is read from FINETUNE_CFG; banner mirrors that
    try:
        with open(FINETUNE_CFG, "r") as f:
            _max = yaml.safe_load(f)["training"]["max_steps"]
        banner(f"Stage 10: Finetune ({_max:,} steps)")
    except Exception:
        banner("Stage 10: Finetune")

    finetune_ckpt = CKPT_DIR / "finetune_1.1_final.pt"
    if finetune_ckpt.exists():
        print(f"[skip] Finetune checkpoint exists: {finetune_ckpt}")
        return

    pretrain_ckpt = CKPT_DIR / "pretrain_1.1_final.pt"
    if not pretrain_ckpt.exists():
        print("ERROR: No pretrain checkpoint!")
        sys.exit(1)

    ft_ckpts = sorted(CKPT_DIR.glob("1.1_ft_step_*.pt"), key=_ckpt_step)
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

    # train.py writes finetune_1.1_final.pt directly. fall back to latest
    # ft step checkpoint only if the run was interrupted.
    if finetune_ckpt.exists():
        print(f"\nFinetune checkpoint: {finetune_ckpt}")
    else:
        ft_ckpts = sorted(CKPT_DIR.glob("1.1_ft_step_*.pt"), key=_ckpt_step)
        if ft_ckpts:
            shutil.copy2(str(ft_ckpts[-1]), str(finetune_ckpt))
            print(f"\nFinetune did not complete; using latest: "
                  f"{ft_ckpts[-1].name} -> {finetune_ckpt.name}")


# ── stage 10: test ───────────────────────────────────────────────────
def stage_test():
    banner("Stage 11: Test")

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
        "verify", "cleanup", "download", "classifiers", "quality", "dedup",
        "tokenizer", "shards", "pretrain", "download_instruct", "synthetic",
        "instruct", "finetune", "test",
    ], help="Run a specific stage")
    args = parser.parse_args()

    print("=" * 64)
    print("  Plasma 1.1 training pipeline")
    print("=" * 64)

    stages = {
        "verify":           stage_verify,
        "cleanup":          stage_cleanup,
        "download":         stage_download,
        "classifiers":      stage_train_classifiers,
        "quality":          stage_quality_score,
        "dedup":            stage_minhash_dedup,
        "tokenizer":        stage_train_tokenizer,
        "shards":           stage_mix_and_shard,
        "pretrain":         stage_pretrain,
        "download_instruct": stage_download_instruct,
        "synthetic":        stage_generate_synthetic,
        "instruct":         stage_build_instruct_shards,
        "finetune":         stage_finetune,
        "test":             stage_test,
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
