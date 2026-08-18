#!/usr/bin/env python3
# robust status monitor for the Plasma 1.1 rebuild.
# auto-detects stage, reports latest step/loss/val/ETA, flags NaN/spike/stall.
# usage: python scripts/monitor_v2.py [--watch SECONDS]

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PRETRAIN_SHARDS = ROOT / "data" / "shards_1.1_v2" / "meta.yaml"
SFT_SHARDS = ROOT / "data" / "sft_shards_1.1_v2" / "meta.yaml"
PRETRAIN_FINAL = ROOT / "checkpoints" / "pretrain_1.1_v2_final.pt"
FINETUNE_FINAL = ROOT / "checkpoints" / "finetune_1.1_v2_final.pt"


def tail_jsonl(path, n=400):
    if not Path(path).exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f.readlines()[-n:]:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def last_where(rows, key):
    for r in reversed(rows):
        if key in r:
            return r
    return None


def fmt_eta(sec):
    if sec is None or sec <= 0:
        return "?"
    d, sec = divmod(int(sec), 86400)
    h, sec = divmod(sec, 3600)
    m, _ = divmod(sec, 60)
    return f"{d}d {h}h {m}m" if d else f"{h}h {m}m"


def gpu_status():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        u, mu, mt = [x.strip() for x in out.stdout.strip().split(",")]
        return f"GPU {u}% util, {mu}/{mt} MiB"
    except Exception:
        return "GPU n/a"


def file_age(path):
    p = Path(path)
    return time.time() - p.stat().st_mtime if p.exists() else None


def report():
    print("=" * 60)
    print(f"Plasma 1.1 rebuild status  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {gpu_status()}")

    if not PRETRAIN_SHARDS.exists():
        log = ROOT / "logs" / "build_corpus_v2.log"
        age = file_age(log)
        lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()[-1:] if log.exists() else []
        print("  STAGE: building pretrain corpus")
        print(f"    {lines[0].strip() if lines else 'no log yet'}")
        if age and age > 600:
            print(f"    ALERT: corpus log idle {int(age)}s")
        print("=" * 60)
        return

    if not PRETRAIN_FINAL.exists():
        stage, jsonl, cfg = "PRETRAIN", "logs/pretrain_v2.jsonl", ROOT / "configs" / "pretrain_1.1_v2.yaml"
    elif not SFT_SHARDS.exists():
        print("  STAGE: building sft shards"); print("=" * 60); return
    elif not FINETUNE_FINAL.exists():
        stage, jsonl, cfg = "FINETUNE", "logs/finetune_v2.jsonl", ROOT / "configs" / "finetune_1.1_v2.yaml"
    else:
        print("  STAGE: COMPLETE — finetune_1.1_v2_final.pt exists. Run scripts/benchmark_v2.py")
        print("=" * 60); return

    max_steps = yaml.safe_load(cfg.read_text(encoding="utf-8"))["training"]["max_steps"]
    rows = tail_jsonl(ROOT / jsonl)
    tr = last_where(rows, "loss")
    ev = last_where(rows, "val_loss")
    age = file_age(ROOT / jsonl)

    print(f"  STAGE: {stage}  (max_steps {max_steps:,})")
    if tr:
        step = tr["step"]
        tps = tr.get("tok_per_sec", 0)
        pct = 100 * step / max_steps
        # ~32768 tok/step pretrain (b32*1024); ~32768 finetune too (b32*1024 effective)
        remaining = (max_steps - step) * 32768 / tps if tps else None
        print(f"    step {step:,}/{max_steps:,} ({pct:.1f}%) | loss {tr['loss']} | "
              f"lr {tr.get('lr')} | grad_norm {tr.get('grad_norm')} | {tps:,} tok/s")
        print(f"    ETA: {fmt_eta(remaining)}")
        loss = tr["loss"]
        if loss != loss:
            print("    ALERT: loss is NaN")
        elif loss > 20:
            print(f"    ALERT: loss {loss} > 20 (diverging)")
        if tr.get("grad_norm", 0) and tr["grad_norm"] > 50:
            print(f"    ALERT: grad_norm {tr['grad_norm']} spiking")
    else:
        print("    no training rows logged yet")
    if ev:
        print(f"    last eval: step {ev['step']} | val_loss {ev['val_loss']} | val_ppl {ev.get('val_ppl')}")
    if age and age > 900:
        print(f"    ALERT: training log idle {int(age)}s — process may be dead/stalled")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="loop every N seconds")
    args = ap.parse_args()
    if args.watch:
        while True:
            report()
            time.sleep(args.watch)
    else:
        report()


if __name__ == "__main__":
    main()
