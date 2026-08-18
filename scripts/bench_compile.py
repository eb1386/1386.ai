#!/usr/bin/env python3
# settle the torch.compile question properly.
#
# whole-model compile measured 1.27-1.41x on a FIXED synthetic tensor but never
# reached a steady state in the real training loop. this tests the modes that
# matter, with real dataloader-style fresh tensors, and reports compile time,
# steady-state throughput and dynamo recompile counts:
#
#   none    - eager baseline
#   model   - torch.compile(model)
#   block   - compile each TransformerBlock (ONE graph reused 26x, and it composes
#             with activation checkpointing instead of fighting it)
#
# run one mode per process so a bad compile cannot poison the others.

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def child(args):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT))
    from src.model.config import ModelConfig
    from src.model.transformer import Transformer
    from src.train.utils import load_config

    cfg = load_config(args.config)
    mc = ModelConfig.from_dict(cfg["model"])
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    model = Transformer(mc).to(dev)
    model.gradient_checkpointing = True
    model.train()

    if args.mode == "model":
        fwd = torch.compile(model)
    elif args.mode == "block":
        for i, layer in enumerate(model.layers):
            model.layers[i] = torch.compile(layer)
        fwd = model
    else:
        fwd = model

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    MB, SL, ACC = args.micro_batch, args.seq_len, args.accum

    def step_once():
        opt.zero_grad(set_to_none=True)
        for _ in range(ACC):
            # fresh tensors every micro-step, exactly like the dataloader
            x = torch.randint(0, mc.vocab_size, (MB, SL), device=dev)
            y = torch.randint(0, mc.vocab_size, (MB, SL), device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = fwd(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       y.reshape(-1))
            (loss / ACC).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()

    t0 = time.time()
    step_once()
    first = time.time() - t0          # includes all compilation

    times = []
    for _ in range(args.steps):
        t1 = time.time()
        step_once()
        times.append(time.time() - t1)

    recompiles = -1
    try:
        import torch._dynamo as dynamo
        recompiles = int(sum(dynamo.utils.counters["frames"].values())) \
            if dynamo.utils.counters.get("frames") else 0
    except Exception:
        pass

    steady = sorted(times)[len(times) // 2]
    print("RESULT " + json.dumps({
        "mode": args.mode,
        "first_step_s": round(first, 1),
        "steady_s_per_step": round(steady, 3),
        "steady_tok_s": round(MB * SL * ACC / steady),
        "all_step_s": [round(t, 2) for t in times],
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "dynamo_frames": recompiles,
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ablation_base.yaml")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--modes", default="none,block,model")
    ap.add_argument("--out", default="logs/bench_compile.json")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--mode", default="none")
    args = ap.parse_args()

    if args.child:
        child(args)
        return

    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # unsupported on this platform
    results = []
    print(f"{'mode':<8}{'1st step(s)':>13}{'steady s/step':>15}{'tok/s':>11}{'peakGB':>9}")
    print("-" * 58)
    for mode in args.modes.split(","):
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "bench_compile.py"), "--child",
               "--mode", mode, "--config", args.config,
               "--micro-batch", str(args.micro_batch), "--accum", str(args.accum),
               "--seq-len", str(args.seq_len), "--steps", str(args.steps)]
        try:
            p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                               text=True, timeout=2400)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line:
                r = json.loads(line[len("RESULT "):])
                results.append(r)
                print(f"{r['mode']:<8}{r['first_step_s']:>13.1f}"
                      f"{r['steady_s_per_step']:>15.3f}{r['steady_tok_s']:>11,}{r['peak_gb']:>9.2f}")
                print(f"        per-step: {r['all_step_s']}")
            else:
                tail = (p.stderr or "").strip().splitlines()[-1:] or ["(no output)"]
                print(f"{mode:<8}  FAILED rc={p.returncode}: {tail[0][:80]}")
        except subprocess.TimeoutExpired:
            print(f"{mode:<8}  TIMEOUT (>40min)")

    if results:
        base = next((r for r in results if r["mode"] == "none"), None)
        best = max(results, key=lambda r: r["steady_tok_s"])
        print("\nBEST:", best["mode"], f"{best['steady_tok_s']:,} tok/s")
        if base and base["steady_tok_s"]:
            print(f"speedup vs eager: {best['steady_tok_s']/base['steady_tok_s']:.2f}x "
                  f"(compile cost {best['first_step_s']:.0f}s once)")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
