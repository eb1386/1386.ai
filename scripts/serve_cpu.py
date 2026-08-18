#!/usr/bin/env python3
# temporary cpu-only chat server.
#
# the gpu is saturated by sft training (~15.5/16.3 GB), so loading the model
# there would OOM the run. this hides cuda before torch imports, caps threads
# so the dataloader workers keep their cores, and registers the newest sft
# checkpoint as a preview model alongside 1.0/1.1.
#
#   python scripts/serve_cpu.py --port 8000

import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# must happen before torch loads
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CKPT_DIR = ROOT / "checkpoints"
# a checkpoint younger than this may still be mid-write
SETTLE_SECONDS = 120


def newest_checkpoint(prefix):
    """newest fully-written checkpoint matching prefix_<step>.pt"""
    found = []
    for p in CKPT_DIR.glob(f"{prefix}*.pt"):
        tail = p.stem[len(prefix):]
        if not tail.isdigit():
            continue
        if time.time() - p.stat().st_mtime < SETTLE_SECONDS:
            print(f"  skipping {p.name}, still being written")
            continue
        found.append((int(tail), p))
    if not found:
        return None, None
    step, path = max(found)
    return step, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=4,
                    help="cpu threads for inference, leave some for the dataloader")
    ap.add_argument("--max-tokens", type=int, default=160,
                    help="shorter than the gpu default, cpu decode is slower")
    ap.add_argument("--no-preview", action="store_true",
                    help="skip registering the in-progress sft checkpoint")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import torch

    # refuse to start if cuda leaked through, training's vram is not ours
    if torch.cuda.is_available():
        raise SystemExit("cuda is still visible, refusing to start")
    torch.set_num_threads(args.threads)

    from web.model_manager import MODEL_REGISTRY

    step, ckpt = (None, None) if args.no_preview else newest_checkpoint("1.1_v3_ft_step_")
    if ckpt:
        MODEL_REGISTRY["plasma-1.1-preview"] = {
            "name": f"Plasma 1.1 — SFT step {step:,} (preview)",
            "config": ROOT / "configs" / "finetune_1.1_v3.yaml",
            "checkpoint": ckpt,
            "tokenizer": ROOT / "data" / "tokenizer_1.1.model",
            "params": "521M",
            "multiturn": True,
        }

    import uvicorn
    import web.app as webapp

    webapp.manager.device = torch.device("cpu")

    # cpu decode is slow, so cap length unless the caller asks for more
    inner = webapp.manager.generate

    def generate(model_id, prompt, **kw):
        kw.setdefault("max_tokens", args.max_tokens)
        t0 = time.time()
        out = inner(model_id, prompt, **kw)
        print(f"  generated {len(out)} chars in {time.time()-t0:.1f}s", flush=True)
        return out

    webapp.manager.generate = generate

    print("\n  1386.ai — cpu only, gpu untouched")
    print(f"  threads {args.threads} | max_tokens {args.max_tokens}")
    for info in MODEL_REGISTRY.values():
        mark = "ready" if info["checkpoint"].exists() else "missing"
        print(f"    {info['name']} ({info['params']}) {mark}")
    print(f"\n  http://localhost:{args.port}\n", flush=True)

    if not args.no_open:
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    uvicorn.run(webapp.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
