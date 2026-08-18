#!/usr/bin/env python3
# throughput benchmark for plasma training configs.
# measures tokens/sec and peak VRAM across micro_batch / grad-checkpointing /
# torch.compile combos so we pick the fastest safe setup.
#
# each config runs in its OWN process: a hard CUDA OOM poisons the cuda context,
# so in-process looping makes every later config fail spuriously.

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_single(args):
    """child mode: benchmark one config, print one json line."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT))
    from src.model.config import ModelConfig
    from src.model.transformer import Transformer
    from src.train.utils import load_config

    cfg = load_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    model_cfg.max_seq_len = max(model_cfg.max_seq_len, args.seq_len)

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = Transformer(model_cfg).to(device)
    model.gradient_checkpointing = args.grad_ckpt
    model.train()
    fwd = torch.compile(model) if args.compile else model

    if args.adam8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                                  weight_decay=0.1)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                                weight_decay=0.1, fused=True)
    x = torch.randint(0, model_cfg.vocab_size, (args.micro_batch, args.seq_len), device=device)
    y = torch.randint(0, model_cfg.vocab_size, (args.micro_batch, args.seq_len), device=device)

    accum = max(1, args.global_batch // args.micro_batch)
    tokens_per_step = args.micro_batch * args.seq_len * accum
    t_start = None
    for step in range(args.steps):
        if step == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t_start = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = fwd(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t_start
    n = args.steps - args.warmup
    print("RESULT " + json.dumps({
        "micro_batch": args.micro_batch, "accum": accum, "grad_ckpt": args.grad_ckpt,
        "compile": args.compile, "adam8bit": args.adam8bit, "seq_len": args.seq_len,
        "tok_s": round(tokens_per_step * n / dt),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "s_per_step": round(dt / n, 3),
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_1.1_v2.yaml")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--out", default="logs/bench_throughput.json")
    ap.add_argument("--micro-batches", default="2,4,8,16")
    ap.add_argument("--try-compile", action="store_true")
    # child-mode flags
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--adam8bit", action="store_true")
    ap.add_argument("--combos", default=None,
                    help="explicit combos: mb:ckpt:compile:adam8bit,... e.g. 2:1:1:0,2:0:0:1")
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    results = []
    combos = []
    if args.combos:
        for spec in args.combos.split(","):
            mb, ck, co, a8 = spec.split(":")
            combos.append((int(mb), ck == "1", co == "1", a8 == "1"))
    else:
        for mb in [int(v) for v in args.micro_batches.split(",")]:
            for ckpt in (True, False):
                combos.append((mb, ckpt, False, False))
                if args.try_compile:
                    combos.append((mb, ckpt, True, False))

    print(f"global batch {args.global_batch} seqs "
          f"({args.global_batch*args.seq_len:,} tok/optimizer step), seq_len {args.seq_len}")
    print(f"{'micro_bs':>8} {'accum':>6} {'gradckpt':>9} {'compile':>8} {'adam8':>6} {'tok/s':>10} {'peakGB':>8}  status")
    print("-" * 80)
    for mb, ckpt, comp, a8 in combos:
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "bench_throughput.py"), "--single",
               "--config", args.config, "--seq-len", str(args.seq_len),
               "--steps", str(args.steps), "--warmup", str(args.warmup),
               "--global-batch", str(args.global_batch), "--micro-batch", str(mb)]
        if ckpt:
            cmd.append("--grad-ckpt")
        if comp:
            cmd.append("--compile")
        if a8:
            cmd.append("--adam8bit")
        accum = max(1, args.global_batch // mb)
        try:
            p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                               text=True, timeout=900)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line:
                r = json.loads(line[len("RESULT "):])
                results.append(r)
                print(f"{mb:>8} {accum:>6} {str(ckpt):>9} {str(comp):>8} {str(a8):>6} "
                      f"{r['tok_s']:>10,} {r['peak_gb']:>8.2f}  ok")
            else:
                err = (p.stderr or "").lower()
                status = "OOM" if "out of memory" in err else f"ERR rc={p.returncode}"
                tail = "" if status == "OOM" else " | " + (p.stderr or "").strip().splitlines()[-1][:60] if p.stderr else ""
                print(f"{mb:>8} {accum:>6} {str(ckpt):>9} {str(comp):>8} {str(a8):>6} {'-':>10} {'-':>8}  {status}{tail}")
        except subprocess.TimeoutExpired:
            print(f"{mb:>8} {accum:>6} {str(ckpt):>9} {str(comp):>8} {str(a8):>6} {'-':>10} {'-':>8}  TIMEOUT")

    if results:
        best = max(results, key=lambda r: r["tok_s"])
        base = next((r for r in results if r["micro_batch"] == 2 and r["grad_ckpt"] and not r["compile"]), None)
        print("\nBEST:", json.dumps(best))
        if base:
            print(f"speedup vs current (mb2+ckpt): {best['tok_s']/base['tok_s']:.2f}x")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": results, "best": best}, f, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
