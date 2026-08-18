#!/usr/bin/env python3
# ONE COMMAND to build and train Plasma 1.1 (v3 recipe), resume-safe.
#
#   python scripts/run_plasma_v3.py
#
# stages, each skipped if its output already exists:
#   1 build pretrain corpus   -> data/shards_1.1_v3
#   2 global shuffle          -> data/shards_1.1_v3s     (kills domain drift)
#   3 validate shards
#   4 pretrain (WSD)          -> checkpoints/pretrain_1.1_v3_final.pt   [auto-resumes]
#   5 build SFT set           -> data/sft_shards_1.1_v3
#   6 finetune                -> checkpoints/finetune_1.1_v3_final.pt   [auto-resumes]
#   7 benchmark vs 1.0 / old 1.1 (custom + standard multiple-choice)
#
# safe to re-run after any crash: pretrain/finetune resume from the newest
# step checkpoint and every build stage is idempotent.

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CKPT = ROOT / "checkpoints"

RAW_SHARDS = ROOT / "data" / "shards_1.1_v4"
SHUF_SHARDS = ROOT / "data" / "shards_1.1_v3s"
SFT_SHARDS = ROOT / "data" / "sft_shards_1.1_v3"
PRETRAIN_CFG = ROOT / "configs" / "pretrain_1.1_v4.yaml"   # balanced mix from step ~185k
FINETUNE_CFG = ROOT / "configs" / "finetune_1.1_v3.yaml"
PRETRAIN_FINAL = CKPT / "pretrain_1.1_v3_final.pt"
FINETUNE_FINAL = CKPT / "finetune_1.1_v3_final.pt"


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"stage failed (exit {r.returncode})")


def latest_step_ckpt(prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.pt$")
    best, best_step = None, -1
    for p in CKPT.glob(f"{prefix}_*.pt"):
        m = pat.match(p.name)
        if m and int(m.group(1)) > best_step:
            best, best_step = p, int(m.group(1))
    return best


def validate(meta_path, min_tokens):
    meta = yaml.safe_load(Path(meta_path).read_text(encoding="utf-8"))
    st = meta.get("source_tokens", {})
    assert meta.get("total_tokens", 0) >= min_tokens, \
        f"corpus too small: {meta.get('total_tokens'):,} < {min_tokens:,}"
    assert not meta.get("instruct_in_pretrain", False), "instruct leaked into pretrain"
    assert meta.get("validation_split") == "document_level", "val split not document-level"
    for k in st:
        assert not str(k).startswith("1.0"), f"old 1.0 contamination: {k}"
    for need in ("code", "math"):
        assert st.get(need, 0) > 0, f"corpus is missing {need} tokens"
    total = max(1, meta["total_tokens"])
    mix = "  ".join(f"{k}={v/total:.0%}" for k, v in sorted(st.items()))
    print(f"[validate] {meta['total_tokens']:,} tokens, "
          f"{meta.get('n_train_sequences',0):,} seqs, shuffled="
          f"{meta.get('globally_shuffled', False)}\n           {mix}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=15_000_000_000)
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--shuffle", action="store_true",
                    help="run an extra global sequence shuffle (usually unnecessary)")
    args = ap.parse_args()

    # 1. pretrain corpus
    if not (RAW_SHARDS / "meta.yaml").exists():
        run([PY, "-u", "scripts/build_corpus_v3.py",
             "--target-tokens", args.target_tokens, "--force"])
    else:
        print("[skip] pretrain corpus exists")

    # 2. optional global shuffle. NOT needed by default: build_corpus_v3 epochs each
    # source so the mixture already holds shard-to-shard (measured: code stays
    # 21-28% from shard 0 through shard 145). use --shuffle if you change the
    # builder in a way that can reintroduce ordering drift.
    corpus = RAW_SHARDS
    if args.shuffle:
        if not (SHUF_SHARDS / "meta.yaml").exists():
            run([PY, "-u", "scripts/shuffle_shards.py",
                 "--in-dir", RAW_SHARDS, "--out-dir", SHUF_SHARDS, "--split", "train"])
        else:
            print("[skip] shuffled corpus exists")
        corpus = SHUF_SHARDS

    # 3. validate
    validate(corpus / "meta.yaml", min_tokens=5_000_000_000)

    # 4. pretrain
    if not PRETRAIN_FINAL.exists():
        cmd = [PY, "-u", "-m", "src.train.train", "--config", PRETRAIN_CFG,
               "--log-path", "logs/pretrain_v3.jsonl"]
        resume = latest_step_ckpt(yaml.safe_load(PRETRAIN_CFG.read_text(encoding="utf-8"))
                                  ["training"]["checkpoint_prefix"])
        if resume:
            print(f"[resume] pretrain from {resume.name}")
            cmd += ["--resume", str(resume)]
        run(cmd)
    else:
        print("[skip] pretrain final exists")

    # 5. SFT data
    if not (SFT_SHARDS / "meta.yaml").exists():
        run([PY, "-u", "scripts/build_sft_v3.py", "--force"])
    else:
        print("[skip] sft shards exist")

    # 6. finetune
    if not FINETUNE_FINAL.exists():
        cmd = [PY, "-u", "-m", "src.train.train", "--config", FINETUNE_CFG,
               "--log-path", "logs/finetune_v3.jsonl"]
        resume = latest_step_ckpt(yaml.safe_load(FINETUNE_CFG.read_text(encoding="utf-8"))
                                  ["training"]["checkpoint_prefix"])
        if resume:
            print(f"[resume] finetune from {resume.name}")
            cmd += ["--resume", str(resume)]
        else:
            cmd += ["--finetune", str(PRETRAIN_FINAL)]
        run(cmd)
    else:
        print("[skip] finetune final exists")

    # 7. benchmarks
    if not args.skip_bench:
        run([PY, "-u", "scripts/benchmark_v2.py",
             "--add", f"1.1_v3:{FINETUNE_FINAL}:{FINETUNE_CFG}",
             "--out", "logs/benchmark_v3.json"])
        run([PY, "-u", "scripts/eval_standard.py",
             "--model", "1.0:checkpoints/finetune_1.0_final.pt:configs/finetune_1.0.yaml",
             "--model", f"1.1_v3:{FINETUNE_FINAL}:{FINETUNE_CFG}",
             "--out", "logs/eval_standard_v3.json"])
    print("\n=== plasma v3 pipeline complete ===")


if __name__ == "__main__":
    main()
