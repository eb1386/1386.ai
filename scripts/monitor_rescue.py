#!/usr/bin/env python3
# Hourly monitor for the Plasma 1.1 rescue SFT run.

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "train.jsonl"
MONITOR_LOG = ROOT / "logs" / "rescue_monitor.log"
FINAL_CKPT = ROOT / "checkpoints" / "finetune_1.1_rescue_final.pt"
EVAL_JSON = ROOT / "logs" / "eval_rescue_final.json"


def read_rescue_segment():
    if not LOG.exists():
        return [], []
    rows = []
    vals = []
    last_step = -1
    current_rows = []
    current_vals = []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        step = d.get("step")
        if step is None:
            continue
        if last_step >= 0 and step < last_step:
            current_rows = []
            current_vals = []
        if "loss" in d:
            current_rows.append(d)
        if "val_loss" in d:
            current_vals.append(d)
        last_step = step
    rows = current_rows
    vals = current_vals
    return rows, vals


def gpu_status():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception as e:
        return f"gpu_status_error={e}"


def log(msg):
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(MONITOR_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_final_eval():
    if EVAL_JSON.exists():
        return
    log("final checkpoint found; running strict rescue eval")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_rescue.py"),
        "--checkpoint",
        str(FINAL_CKPT),
        "--config",
        str(ROOT / "configs" / "finetune_1.1_rescue.yaml"),
        "--device",
        "cpu",
        "--json-out",
        str(EVAL_JSON),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    log("eval stdout:\n" + result.stdout.strip())
    if result.stderr.strip():
        log("eval stderr:\n" + result.stderr.strip())
    log(f"eval exit code: {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--max-steps", type=int, default=6000)
    args = parser.parse_args()

    log("rescue monitor started")
    while True:
        rows, vals = read_rescue_segment()
        if rows:
            latest = rows[-1]
            remaining = max(0, args.max_steps - int(latest["step"]))
            msg = (
                f"step={latest['step']}/{args.max_steps} "
                f"loss={latest.get('loss')} lr={latest.get('lr')} "
                f"tok_s={latest.get('tok_per_sec')} remaining={remaining} "
                f"gpu={gpu_status()}"
            )
            if vals:
                msg += f" latest_val={vals[-1].get('val_loss')}@{vals[-1].get('step')}"
            log(msg)
        else:
            log(f"no rescue training rows yet gpu={gpu_status()}")

        if FINAL_CKPT.exists():
            run_final_eval()
            log("rescue monitor complete")
            return

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
