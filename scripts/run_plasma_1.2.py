#!/usr/bin/env python3
# ONE COMMAND to build and train Plasma 1.2, resume-safe.
#
#   python scripts/run_plasma_1.2.py            # run everything
#   python scripts/run_plasma_1.2.py --status   # show stage states, do nothing
#   python scripts/run_plasma_1.2.py --until 3  # stop after stage 3 (data prep
#                                               # can run before committing gpu)
#
# stages, each skipped if its output already exists:
#   1 download raw data        -> data/raw_1.2/*.done          (network, hours)
#   2 train tokenizer          -> data/tokenizer_1.2.model     (cpu, ~30 min)
#   3 build 30B corpus         -> data/shards_1.2/meta.yaml    (cpu, ~10-14 h)
#   4 build anneal corpus      -> data/shards_1.2_anneal/      (cpu, ~1-2 h)
#   5 pretrain (wsd, 90%)      -> checkpoints/pretrain_1.2_final.pt  [resumes]
#       swap to anneal shards for the final 10% happens via config handoff,
#       same recipe that worked for 1.1
#   6 build sft shards         -> data/sft_shards_1.2/meta.yaml (cpu, ~20 min)
#   7 finetune                 -> checkpoints/finetune_1.2_final.pt   [resumes]
#   8 benchmark vs 1.1 / 1.0   -> logs/eval_1.2.json
#
# HONEST WALL-CLOCK for stage 5 at measured 1.1 throughput scaled by flops:
# 28-36 days for 30B tokens on the 5080. wsd means an early cooldown at any
# point still yields a usable model (20B tokens ~= 3 weeks).

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STAGES = [
    ("download raw data", ROOT / "data" / "raw_1.2" / "wiki_full.done",
     [PY, "scripts/download_1.2_data.py"]),
    ("train tokenizer", ROOT / "data" / "tokenizer_1.2.model",
     [PY, "scripts/train_tokenizer_1.2.py"]),
    ("build 30B corpus", ROOT / "data" / "shards_1.2" / "meta.yaml",
     [PY, "scripts/build_corpus_1.2.py", "--mix", "main"]),
    ("build anneal corpus", ROOT / "data" / "shards_1.2_anneal" / "meta.yaml",
     [PY, "scripts/build_corpus_1.2.py", "--mix", "anneal"]),
    ("pretrain 756M / 30B", ROOT / "checkpoints" / "pretrain_1.2_final.pt",
     [PY, "-u", "-m", "src.train.train", "--config", "configs/pretrain_1.2.yaml",
      "--log-path", "logs/pretrain_1.2.jsonl"]),
    ("build sft shards", ROOT / "data" / "sft_shards_1.2" / "meta.yaml",
     [PY, "scripts/build_sft_v4.py", "--tokenizer", "data/tokenizer_1.2.model",
      "--template", "special", "--seq-len", "2048",
      "--out-dir", "data/sft_shards_1.2"]),
    ("finetune", ROOT / "checkpoints" / "finetune_1.2_final.pt",
     [PY, "-u", "-m", "src.train.train", "--config", "configs/finetune_1.2.yaml",
      "--finetune", "checkpoints/pretrain_1.2_final.pt",
      "--log-path", "logs/finetune_1.2.jsonl"]),
    ("benchmark", ROOT / "logs" / "eval_1.2.json",
     [PY, "scripts/benchmark_v2.py", "--out", "logs/eval_1.2.json",
      "--add", "plasma_1.2:checkpoints/finetune_1.2_final.pt:configs/finetune_1.2.yaml",
      "--add", "plasma_1.1:checkpoints/finetune_1.1_v3_final.pt:configs/finetune_1.1_v3.yaml"]),
]


def resume_args(stage_name):
    """pretrain/finetune resume from their newest step checkpoint"""
    prefix = {"pretrain 756M / 30B": "1.2_step", "finetune": "1.2_ft_step"}.get(stage_name)
    if not prefix:
        return []
    cks = sorted((ROOT / "checkpoints").glob(f"{prefix}_*.pt"),
                 key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return ["--resume", f"checkpoints/{cks[-1].name}"] if cks else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--until", type=int, default=len(STAGES))
    args = ap.parse_args()

    print("plasma 1.2 pipeline")
    print("=" * 62)
    for i, (name, marker, _) in enumerate(STAGES, 1):
        state = "DONE" if marker.exists() else "pending"
        print(f"  {i}. [{state:>7}] {name}")
    if args.status:
        return
    print("=" * 62)
    print("NOTE: stage 5 is a ~28-36 day gpu run. it auto-resumes from the")
    print("newest checkpoint, and a wsd early-cooldown at any point yields a")
    print("usable model. ctrl+c is always safe.\n")

    for i, (name, marker, cmd) in enumerate(STAGES, 1):
        if i > args.until:
            print(f"\nstopping before stage {i} (--until {args.until})")
            return
        if marker.exists():
            print(f"[{i}/{len(STAGES)}] {name}: done, skipping")
            continue
        full = cmd + resume_args(name)
        print(f"\n[{i}/{len(STAGES)}] {name}\n  $ {' '.join(full)}", flush=True)
        r = subprocess.run(full, cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit(f"stage '{name}' failed with code {r.returncode}; "
                             f"re-run this script to retry from here")
        if not marker.exists():
            raise SystemExit(f"stage '{name}' exited 0 but {marker} is missing")
    print("\nPLASMA 1.2 PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
