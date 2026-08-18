#!/usr/bin/env python3
# controlled ablation runner.
#
# runs short training jobs that differ in exactly one (or a few) config values,
# holding data/seed/steps fixed, then reports final val loss side by side.
# each variant runs in its own process so a CUDA OOM cannot poison the others.
#
# usage:
#   python scripts/ablate.py --name lr_sweep --steps 1500 \
#       --vary training.learning_rate=1.2e-4,3e-4,6e-4,1e-3
#   python scripts/ablate.py --name init --steps 1500 \
#       --vary training.depth_scaled_init=false,true

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "runs" / "ablations"


def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def coerce(s):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        return float(s)
    except ValueError:
        return s


def read_val(jsonl_path):
    """return (last_val_loss, last_train_loss, last_step)."""
    val, tr, step = None, None, 0
    if not Path(jsonl_path).exists():
        return val, tr, step
    for line in Path(jsonl_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "val_loss" in r:
            val = r["val_loss"]
        if "loss" in r:
            tr = r["loss"]
            step = r.get("step", step)
    return val, tr, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-config", default="configs/pretrain_1.1_v2.yaml")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--vary", action="append", required=True,
                    help="dotted.key=v1,v2,v3 (repeatable; multiple keys are zipped)")
    ap.add_argument("--shard-dir", default=None)
    ap.add_argument("--timeout", type=int, default=36000)
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    specs = []
    for v in args.vary:
        key, vals = v.split("=", 1)
        specs.append((key, [coerce(x) for x in vals.split(",")]))
    n_variants = max(len(v) for _, v in specs)

    out_dir = ABL / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"ablation '{args.name}': {n_variants} variants x {args.steps} steps")
    for key, vals in specs:
        print(f"  vary {key}: {vals}")
    print()

    for i in range(n_variants):
        cfg = copy.deepcopy(base)
        label_bits = []
        for key, vals in specs:
            val = vals[i % len(vals)]
            set_path(cfg, key, val)
            label_bits.append(f"{key.split('.')[-1]}={val}")
        label = " ".join(label_bits)
        tag = f"{args.name}_{i}"

        cfg["training"]["max_steps"] = args.steps
        cfg["training"]["warmup_steps"] = args.warmup if args.warmup is not None else max(50, args.steps // 20)
        cfg["training"]["eval_interval"] = args.eval_interval or max(100, args.steps // 3)
        cfg["training"]["checkpoint_interval"] = 10 ** 9  # no step checkpoints
        cfg["training"]["keep_last_checkpoints"] = 0
        cfg["training"]["checkpoint_prefix"] = f"abl_{tag}"
        cfg["training"]["log_interval"] = 50
        if args.shard_dir:
            cfg["data"]["shard_dir"] = args.shard_dir

        cfg_path = out_dir / f"{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        log_jsonl = out_dir / f"{tag}.jsonl"
        log_txt = out_dir / f"{tag}.log"
        if log_jsonl.exists():
            try:
                log_jsonl.unlink()
            except OSError as e:
                raise SystemExit(
                    f"{log_jsonl} is locked by another process -- an ablation with "
                    f"name '{args.name}' is probably already running: {e}")

        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cmd = [sys.executable, "-u", "-m", "src.train.train",
               "--config", str(cfg_path), "--log-path", str(log_jsonl)]
        print(f"[{i+1}/{n_variants}] {label} ...", flush=True)
        t0 = time.time()
        with open(log_txt, "w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf,
                               stderr=subprocess.STDOUT, timeout=args.timeout)
        dt = time.time() - t0
        val, tr, step = read_val(log_jsonl)
        ok = p.returncode == 0
        results.append({"label": label, "val_loss": val, "train_loss": tr,
                        "step": step, "seconds": round(dt), "ok": ok})
        print(f"      val_loss={val} train_loss={tr} step={step} "
              f"({dt/60:.1f} min){'' if ok else '  FAILED rc=%d' % p.returncode}", flush=True)

        # clean the final checkpoint this variant wrote
        for pat in (f"pretrain_abl_{tag}_final.pt", f"finetune_abl_{tag}_final.pt"):
            fp = ROOT / "checkpoints" / pat
            if fp.exists():
                fp.unlink()

    print("\n" + "=" * 72)
    print(f"{'variant':<44}{'val_loss':>10}{'train':>9}{'min':>7}")
    print("-" * 72)
    valid = [r for r in results if r["val_loss"] is not None]
    best = min(valid, key=lambda r: r["val_loss"]) if valid else None
    for r in results:
        star = " *" if best and r is best else ""
        v = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "n/a"
        t = f"{r['train_loss']:.4f}" if r["train_loss"] is not None else "n/a"
        print(f"{r['label']:<44}{v:>10}{t:>9}{r['seconds']/60:>7.1f}{star}")
    print("=" * 72)
    if best:
        print(f"BEST: {best['label']}  val_loss={best['val_loss']:.4f}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
