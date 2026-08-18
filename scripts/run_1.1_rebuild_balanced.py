#!/usr/bin/env python3
# Safe end-to-end Plasma 1.1 rebuild pipeline.

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str], log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"Command failed with exit code {code}: {' '.join(cmd)}")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def final_name(prefix: str, finetune: bool) -> str:
    cleaned = prefix.rstrip("_").replace("_step", "").replace("_ft", "")
    return f"finetune_{cleaned}_final.pt" if finetune else f"pretrain_{cleaned}_final.pt"


def latest_step(prefix: str) -> Path | None:
    ckpts = sorted((ROOT / "checkpoints").glob(f"{prefix}_*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def validate_pretrain_shards(shard_dir: Path, min_tokens: int):
    meta_path = shard_dir / "meta.yaml"
    if not meta_path.exists():
        raise SystemExit(f"Missing shard metadata: {meta_path}")
    meta = load_yaml(meta_path)
    if int(meta.get("total_tokens", 0)) < min_tokens:
        raise SystemExit(f"Pretrain shards too small: {meta.get('total_tokens')} < {min_tokens}")
    bad_sources = [s for s in meta.get("source_tokens", {}) if s.startswith("1.0")]
    if bad_sources:
        raise SystemExit(f"Rejected contaminated pretrain sources: {bad_sources}")
    if meta.get("validation_split") != "document_level":
        raise SystemExit("Rejected shards without document-level validation split")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=time.strftime("1.1_rebuild_balanced_%Y%m%d_%H%M%S"))
    parser.add_argument("--target-tokens", type=int, default=7_500_000_000)
    parser.add_argument("--force-shards", action="store_true")
    parser.add_argument("--no-download-missing", action="store_true")
    parser.add_argument("--allow-missing-code-stack", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run_id
    log_dir = run_dir / "logs"
    cfg_dir = run_dir / "configs"
    pretrain_shards = ROOT / "data" / f"shards_{args.run_id}_pretrain"
    sft_shards = ROOT / "data" / f"shards_{args.run_id}_sft"
    run_log = log_dir / "run.log"

    pre_cfg = load_yaml(ROOT / "configs" / "pretrain_1.1_rebuild.yaml")
    pre_prefix = f"{args.run_id}_pretrain_step"
    pre_cfg["data"]["shard_dir"] = str(pretrain_shards.relative_to(ROOT))
    pre_cfg["training"]["checkpoint_prefix"] = pre_prefix
    pre_cfg["training"]["log_path"] = str((log_dir / "pretrain.jsonl").relative_to(ROOT))
    # Match training steps to the actual rebuilt token budget. Each step sees
    # batch_size * seq_len tokens, so this avoids silently doing extra epochs
    # when the local corpus is smaller than the ideal Chinchilla target.
    tokens_per_step = int(pre_cfg["training"]["batch_size"]) * int(pre_cfg["data"]["seq_len"])
    derived_steps = max(1, args.target_tokens // tokens_per_step)
    pre_cfg["training"]["max_steps"] = int(derived_steps)
    pre_cfg["training"]["warmup_steps"] = min(int(pre_cfg["training"].get("warmup_steps", 3000)), max(100, int(derived_steps // 100)))
    pre_cfg_path = cfg_dir / "pretrain.yaml"
    write_yaml(pre_cfg_path, pre_cfg)
    pre_final = ROOT / "checkpoints" / final_name(pre_prefix, finetune=False)

    ft_cfg = load_yaml(ROOT / "configs" / "finetune_1.1_balanced_pretrain.yaml")
    ft_prefix = f"{args.run_id}_balanced_ft_step"
    ft_cfg["data"]["shard_dir"] = str(sft_shards.relative_to(ROOT))
    ft_cfg["training"]["checkpoint_prefix"] = ft_prefix
    ft_cfg["training"]["log_path"] = str((log_dir / "finetune.jsonl").relative_to(ROOT))
    ft_cfg_path = cfg_dir / "finetune.yaml"
    write_yaml(ft_cfg_path, ft_cfg)
    ft_final = ROOT / "checkpoints" / final_name(ft_prefix, finetune=True)

    manifest = {
        "run_id": args.run_id,
        "pretrain_shards": str(pretrain_shards),
        "sft_shards": str(sft_shards),
        "pretrain_config": str(pre_cfg_path),
        "finetune_config": str(ft_cfg_path),
        "pretrain_final": str(pre_final),
        "finetune_final": str(ft_final),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    force = ["--force"] if args.force_shards else []
    code_path = ROOT / "data" / "raw_1.1" / "code_clean.txt"
    stack_path = ROOT / "data" / "raw_1.1" / "stackexchange_clean.txt"
    missing_domains = [
        p.name for p in (code_path, stack_path)
        if not p.exists() or p.stat().st_size == 0
    ]
    if missing_domains and not args.no_download_missing:
        print(f"Missing broad-domain sources: {', '.join(missing_domains)}")
        run([PY, "-u", "scripts/run_1.1.py", "--stage", "download"], run_log)

    require_domains = [] if args.allow_missing_code_stack else ["--require-code-stack"]
    if not (pretrain_shards / "meta.yaml").exists():
        run(
            [PY, "-u", "scripts/build_repretrain_shards.py", "--out-dir", str(pretrain_shards.relative_to(ROOT)), "--target-tokens", str(args.target_tokens), "--chunk-seqs", "100000", *force, *require_domains],
            run_log,
        )
    validate_pretrain_shards(pretrain_shards, min_tokens=min(args.target_tokens, 1_000_000_000))

    if not args.skip_pretrain and not pre_final.exists():
        resume = latest_step(pre_prefix)
        cmd = [PY, "-u", "-m", "src.train.train", "--config", str(pre_cfg_path.relative_to(ROOT)), "--log-path", str(log_dir / "pretrain.jsonl")]
        if resume:
            cmd.extend(["--resume", str(resume.relative_to(ROOT))])
        run(cmd, run_log)
    if not pre_final.exists() and not args.skip_pretrain:
        raise SystemExit(f"Pretrain did not produce final checkpoint: {pre_final}")

    if not (sft_shards / "meta.yaml").exists():
        run([PY, "-u", "scripts/build_balanced_sft.py", "--out-dir", str(sft_shards.relative_to(ROOT)), *force], run_log)

    if not args.skip_sft and not ft_final.exists():
        source_pretrain = pre_final if pre_final.exists() else ROOT / "checkpoints" / "pretrain_1.1_final.pt"
        if not source_pretrain.exists():
            raise SystemExit(f"Missing pretrain checkpoint for SFT: {source_pretrain}")
        run(
            [PY, "-u", "-m", "src.train.train", "--config", str(ft_cfg_path.relative_to(ROOT)), "--finetune", str(source_pretrain.relative_to(ROOT)), "--log-path", str(log_dir / "finetune.jsonl")],
            run_log,
        )

    if ft_final.exists():
        run([PY, "scripts/eval_rescue.py", "--checkpoint", str(ft_final.relative_to(ROOT)), "--config", str(ft_cfg_path.relative_to(ROOT)), "--device", "cpu", "--json-out", str(log_dir / "eval_exact.json")], run_log)
        run([PY, "scripts/eval_holdout.py", "--checkpoint", str(ft_final.relative_to(ROOT)), "--config", str(ft_cfg_path.relative_to(ROOT)), "--device", "cpu", "--json-out", str(log_dir / "eval_holdout.json")], run_log)

    print("\nPipeline finished.")
    print(f"Run dir: {run_dir}")
    print(f"Pretrain final: {pre_final}")
    print(f"Finetune final: {ft_final}")


if __name__ == "__main__":
    main()
