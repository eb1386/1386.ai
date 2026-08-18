#!/usr/bin/env python3
# ONE COMMAND to watch the Plasma pipeline:
#
#   python scripts/monitor.py --watch 120
#
# auto-detects the current stage, reports real progress numbers and an ETA, and
# flags the failure modes that actually happen: NaN/diverging loss, gradient
# spikes, a dead or stalled process, and val loss regressing.

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PRETRAIN_CFG = ROOT / "configs" / "pretrain_1.1_v3.yaml"
FINETUNE_CFG = ROOT / "configs" / "finetune_1.1_v3.yaml"

STAGES = [
    ("build corpus",  ROOT / "data" / "shards_1.1_v3" / "meta.yaml",  ROOT / "logs" / "build_corpus_v3.log"),
    # the balanced remix the run actually trained on; the old v3s path never
    # existed, so the monitor used to report "not started" on a finished run
    ("balanced corpus", ROOT / "data" / "shards_1.1_v4" / "meta.yaml", ROOT / "logs" / "build_corpus_v4.log"),
    ("pretrain",      ROOT / "checkpoints" / "pretrain_1.1_v3_final.pt", ROOT / "logs" / "pretrain_v3.jsonl"),
    ("build sft",     ROOT / "data" / "sft_shards_1.1_v3" / "meta.yaml", ROOT / "logs" / "build_sft_v3.log"),
    ("finetune",      ROOT / "checkpoints" / "finetune_1.1_v3_final.pt", ROOT / "logs" / "finetune_v3.jsonl"),
]


def tail_jsonl(path, n=600):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def last_with(rows, key):
    for r in reversed(rows):
        if key in r:
            return r
    return None


def fmt_dur(sec):
    if not sec or sec <= 0:
        return "?"
    d, sec = divmod(int(sec), 86400)
    h, sec = divmod(sec, 3600)
    m, _ = divmod(sec, 60)
    return f"{d}d {h}h {m}m" if d else f"{h}h {m}m"


def gpu():
    try:
        o = subprocess.run(["nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15).stdout.strip()
        u, mu, mt, t = [x.strip() for x in o.split(",")]
        return f"GPU {u}% | {mu}/{mt} MiB | {t}C"
    except Exception:
        return "GPU n/a"


def procs():
    try:
        o = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                            "Where-Object { $_.CommandLine -like '*src.train.train*' } | "
                            "Measure-Object).Count"],
                           capture_output=True, text=True, timeout=25).stdout.strip()
        return int(o or 0)
    except Exception:
        return -1


def age(p):
    p = Path(p)
    return time.time() - p.stat().st_mtime if p.exists() else None


def wsd_phase(cfg, step):
    t = cfg["training"]
    if str(t.get("lr_schedule", "cosine")).lower() not in ("wsd", "trapezoid", "trapezoidal"):
        return ""
    ms, wu = t["max_steps"], t["warmup_steps"]
    dec = int(t.get("decay_steps") or ms * float(t.get("decay_frac", 0.1)))
    start = max(wu, ms - dec)
    if step < wu:
        return " [warmup]"
    return " [stable]" if step < start else " [COOLDOWN]"


def report():
    print("=" * 66)
    print(f"Plasma 1.1 v3   {time.strftime('%Y-%m-%d %H:%M:%S')}   {gpu()}")

    stage = None
    for name, done_marker, log in STAGES:
        if not Path(done_marker).exists():
            stage = (name, log)
            break
    if stage is None:
        print("  STAGE: COMPLETE -- run: python scripts/benchmark_v2.py")
        print("=" * 66)
        return

    name, log = stage
    print(f"  STAGE: {name}")

    if name in ("pretrain", "finetune"):
        cfg = yaml.safe_load((PRETRAIN_CFG if name == "pretrain" else FINETUNE_CFG)
                             .read_text(encoding="utf-8"))
        max_steps = cfg["training"]["max_steps"]
        tok_per_step = cfg["training"]["batch_size"] * cfg["data"]["seq_len"]
        rows = tail_jsonl(log)
        tr, ev = last_with(rows, "loss"), last_with(rows, "val_loss")
        n_proc = procs()
        if tr:
            step, tps = tr["step"], tr.get("tok_per_sec", 0) or 0
            remain = (max_steps - step) * tok_per_step / tps if tps else None
            print(f"    step {step:,}/{max_steps:,} ({100*step/max_steps:.1f}%)"
                  f"{wsd_phase(cfg, step)}  loss {tr['loss']}  lr {tr.get('lr')}  "
                  f"|g| {tr.get('grad_norm')}  {tps:,} tok/s")
            print(f"    tokens seen {step*tok_per_step/1e9:.2f}B   ETA {fmt_dur(remain)}")
            loss = tr["loss"]
            if loss != loss:
                print("    *** ALERT: loss is NaN ***")
            elif loss > 20:
                print(f"    *** ALERT: loss {loss} diverging ***")
            if (tr.get("grad_norm") or 0) > 50:
                print(f"    *** ALERT: grad_norm {tr['grad_norm']} spiking ***")
        else:
            print("    no training rows yet")
        vals = [r for r in rows if "val_loss" in r]
        if ev:
            trend = ""
            if len(vals) >= 2 and vals[-1]["val_loss"] > vals[-2]["val_loss"]:
                trend = f"  (UP from {vals[-2]['val_loss']})"
            print(f"    val_loss {ev['val_loss']} ppl {ev.get('val_ppl')} @step {ev['step']}{trend}")
        a = age(log)
        if n_proc == 0:
            print("    *** ALERT: no training process running ***")
        if a and a > 900:
            print(f"    *** ALERT: log idle {int(a)}s -- stalled? ***")
    else:
        p = Path(log)
        if p.exists():
            lines = [l for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
            print(f"    {lines[-1][:110] if lines else '(no output yet)'}")
            a = age(log)
            if a and a > 900:
                print(f"    *** ALERT: log idle {int(a)}s ***")
        else:
            print("    (not started)")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="repeat every N seconds")
    args = ap.parse_args()
    while True:
        report()
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
