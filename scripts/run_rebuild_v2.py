#!/usr/bin/env python3
# one-command Plasma 1.1 clean rebuild orchestrator (resume-safe).
# stages: build corpus -> validate -> pretrain -> build sft -> finetune -> benchmark.
# each stage is skipped if its output already exists; pretrain auto-resumes from the
# latest step checkpoint. safe to re-run after any crash.

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CKPT = ROOT / "checkpoints"

PRETRAIN_CFG = ROOT / "configs" / "pretrain_1.1_v2.yaml"
FINETUNE_CFG = ROOT / "configs" / "finetune_1.1_v2.yaml"
PRETRAIN_SHARDS = ROOT / "data" / "shards_1.1_v2"
SFT_SHARDS = ROOT / "data" / "sft_shards_1.1_v2"
PRETRAIN_FINAL = CKPT / "pretrain_1.1_v2_final.pt"
FINETUNE_FINAL = CKPT / "finetune_1.1_v2_final.pt"


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), **kw)
    if r.returncode != 0:
        raise SystemExit(f"stage failed (exit {r.returncode}): {cmd}")


def latest_step_ckpt(prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.pt$")
    best, best_step = None, -1
    for p in CKPT.glob(f"{prefix}_*.pt"):
        m = pat.match(p.name)
        if m and int(m.group(1)) > best_step:
            best, best_step = p, int(m.group(1))
    return best


def validate_pretrain_shards():
    meta = yaml.safe_load((PRETRAIN_SHARDS / "meta.yaml").read_text(encoding="utf-8"))
    assert meta.get("code_source_present"), "pretrain shards missing code"
    assert meta.get("stackexchange_source_present"), "pretrain shards missing stackexchange"
    assert not meta.get("instruct_in_pretrain", False), "instruct leaked into pretrain"
    assert meta.get("validation_split") == "document_level", "val split not document-level"
    assert meta.get("total_tokens", 0) >= 3_000_000_000, "pretrain corpus too small"
    for k in meta.get("source_tokens", {}):
        assert not str(k).startswith("1.0"), f"old-1.0 source contamination: {k}"
    print(f"[validate] pretrain shards OK: {meta['total_tokens']:,} tokens, "
          f"{meta['n_train_sequences']:,} train seqs", flush=True)


def main():
    # 1. build pretrain corpus
    if not (PRETRAIN_SHARDS / "meta.yaml").exists():
        run([PY, "-u", "scripts/build_pretrain_corpus.py", "--force"])
    else:
        print("[skip] pretrain corpus exists", flush=True)
    validate_pretrain_shards()

    # 2. pretrain (resume-safe)
    if not PRETRAIN_FINAL.exists():
        cmd = [PY, "-u", "-m", "src.train.train", "--config", str(PRETRAIN_CFG),
               "--log-path", "logs/pretrain_v2.jsonl"]
        resume = latest_step_ckpt("1.1_v2_step")
        if resume:
            print(f"[resume] pretrain from {resume.name}", flush=True)
            cmd += ["--resume", str(resume)]
        run(cmd)
    else:
        print("[skip] pretrain final exists", flush=True)

    # 3. build sft
    if not (SFT_SHARDS / "meta.yaml").exists():
        run([PY, "-u", "scripts/build_sft_v2.py", "--force"])
    else:
        print("[skip] sft shards exist", flush=True)

    # 4. finetune (resume-safe)
    if not FINETUNE_FINAL.exists():
        cmd = [PY, "-u", "-m", "src.train.train", "--config", str(FINETUNE_CFG),
               "--log-path", "logs/finetune_v2.jsonl"]
        resume = latest_step_ckpt("1.1_v2_ft_step")
        if resume:
            print(f"[resume] finetune from {resume.name}", flush=True)
            cmd += ["--resume", str(resume)]
        else:
            cmd += ["--finetune", str(PRETRAIN_FINAL)]
        run(cmd)
    else:
        print("[skip] finetune final exists", flush=True)

    # 5. benchmark
    run([PY, "-u", "scripts/benchmark_v2.py"])
    print("\n=== rebuild complete ===", flush=True)


if __name__ == "__main__":
    main()
