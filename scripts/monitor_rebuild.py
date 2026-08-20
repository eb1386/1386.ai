#!/usr/bin/env python3
# Lightweight hourly monitor for the Plasma 1.1 rebuild pipeline.

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, errors="replace", stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"ERROR: {e}"


def tail(path: Path, n: int = 8) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception as e:
        return [f"tail error: {e}"]


def latest_jsonl(path: Path) -> dict | None:
    for line in reversed(tail(path, 200)):
        try:
            return json.loads(line)
        except Exception:
            continue
    return None


def newest_mtime(paths: list[Path]) -> float:
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run_id
    log_dir = run_dir / "logs"
    monitor_log = log_dir / "monitor.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    def write(msg: str):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with open(monitor_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    write("monitor started")
    seen_eval = set()
    while True:
        gpu = run_text(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
        procs = run_text(["powershell", "-NoProfile", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"])
        raw_code = ROOT / "data" / "raw_1.1" / "code_clean.txt"
        raw_stack = ROOT / "data" / "raw_1.1" / "stackexchange_clean.txt"
        code_size = raw_code.stat().st_size if raw_code.exists() else 0
        stack_size = raw_stack.stat().st_size if raw_stack.exists() else 0
        shard_dirs = list((ROOT / "data").glob(f"shards_{args.run_id}_pretrain*"))
        shard_mtime = newest_mtime(shard_dirs)
        pretrain_ckpts = list((ROOT / "checkpoints").glob(f"{args.run_id}_pretrain_step_*.pt"))
        finetune_ckpts = list((ROOT / "checkpoints").glob(f"{args.run_id}_balanced_ft_step_*.pt"))
        pre = latest_jsonl(log_dir / "pretrain.jsonl")
        ft = latest_jsonl(log_dir / "finetune.jsonl")
        last_run = " | ".join(tail(log_dir / "run.log", 3))
        now = time.time()
        python_alive = bool(procs and "ERROR:" not in procs and procs.strip())
        verdict = "OK"
        reasons = []
        if not python_alive:
            verdict = "ALERT"
            reasons.append("no python process visible")
        if code_size == 0 or stack_size == 0:
            verdict = "WAIT"
            reasons.append("missing broad-domain corpus file")
        if shard_dirs and shard_mtime and now - shard_mtime > 7200 and pre is None:
            verdict = "ALERT"
            reasons.append("pretrain shard output stale for >2h")
        if pre and "loss" in pre:
            if not isinstance(pre["loss"], (int, float)) or pre["loss"] != pre["loss"] or pre["loss"] > 20:
                verdict = "ALERT"
                reasons.append(f"bad pretrain loss {pre.get('loss')}")
        if ft and "loss" in ft:
            if not isinstance(ft["loss"], (int, float)) or ft["loss"] != ft["loss"] or ft["loss"] > 20:
                verdict = "ALERT"
                reasons.append(f"bad finetune loss {ft.get('loss')}")
        write(
            f"verdict={verdict} reasons={'; '.join(reasons) if reasons else 'none'} "
            f"gpu={gpu} python_pids={procs or 'none'} "
            f"code_gb={code_size/1e9:.2f} stack_gb={stack_size/1e9:.2f} "
            f"shard_dirs={len(shard_dirs)} pretrain_ckpts={len(pretrain_ckpts)} finetune_ckpts={len(finetune_ckpts)} "
            f"pretrain={pre} finetune={ft} run_tail={last_run}"
        )

        ckpts = sorted((ROOT / "checkpoints").glob(f"{args.run_id}_balanced_ft_step_*.pt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            latest = ckpts[-1]
            if latest.name not in seen_eval:
                seen_eval.add(latest.name)
                out = log_dir / f"eval_holdout_{latest.stem}.json"
                write(f"running holdout eval for {latest.name}")
                cmd = [
                    str(Path.home() / "AppData/Local/Programs/Python/Python310/python.exe"),
                    "scripts/eval_holdout.py",
                    "--checkpoint", str(latest.relative_to(ROOT)),
                    "--config", str((run_dir / "configs" / "finetune.yaml").relative_to(ROOT)),
                    "--device", "cpu",
                    "--json-out", str(out.relative_to(ROOT)),
                ]
                write(run_text(cmd))
                rand_out = log_dir / f"eval_random_{latest.stem}_{int(time.time())}.json"
                seed = int(time.time()) % 1_000_000_000
                write(f"running randomized holdout eval for {latest.name} seed={seed}")
                rand_cmd = [
                    str(Path.home() / "AppData/Local/Programs/Python/Python310/python.exe"),
                    "scripts/eval_random_holdout.py",
                    "--checkpoint", str(latest.relative_to(ROOT)),
                    "--config", str((run_dir / "configs" / "finetune.yaml").relative_to(ROOT)),
                    "--device", "cpu",
                    "--seed", str(seed),
                    "--json-out", str(rand_out.relative_to(ROOT)),
                ]
                write(run_text(rand_cmd))

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
