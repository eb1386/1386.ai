#!/usr/bin/env python3
# cpu battery runner
#
# generates model responses for the audit battery under named sampling
# conditions. cpu only, below-normal priority, resumable (skips ids already
# in the output file), shardable across processes.
#
#   python scripts/audit/run_battery.py --subset grid --conditions cur,rp11,rp10,t03,greedy,greedy_rp --shard 0/2 --out logs/audit/grid.jsonl

import argparse
import json
import os
import sys
import time
from pathlib import Path

# cpu only
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# stay polite while the user games
try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # below normal
except Exception:
    pass

import torch  # noqa: E402
import sentencepiece as spm  # noqa: E402

from scripts.audit.battery_prompts import battery, grid_subset  # noqa: E402
from src.model.config import ModelConfig  # noqa: E402
from src.model.transformer import Transformer  # noqa: E402
from src.train.utils import load_config  # noqa: E402
from src.inference.generate import generate  # noqa: E402
from src.inference.template import build_prompt_ids, stop_sequences, penalty_exclude  # noqa: E402

# sampling conditions under test. cur = the pre-audit serving stack.
# seam = token-level prompt building (training-matched), plain = string encode.
CONDITIONS = {
    "cur":       dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.3),
    "rp11":      dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.1),
    "rp10":      dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.0),
    "t03":       dict(temperature=0.3, top_k=40, top_p=0.90, repetition_penalty=1.1),
    "greedy":    dict(temperature=0.0, top_k=0,  top_p=1.0,  repetition_penalty=1.0),
    "greedy_rp": dict(temperature=0.0, top_k=0,  top_p=1.0,  repetition_penalty=1.15),
    # audit-proposed serving stack: seam fix + gentle penalty + stop seqs
    "fixed":     dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.1,
                      penalty_window=128, seam=True, stops=True, exclude=True),
    # seam fix alone, old params: isolates the tokenization effect
    "seam":      dict(temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.3,
                      seam=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/finetune_1.1_v3_final.pt")
    ap.add_argument("--config", default="configs/finetune_1.1_v3.yaml")
    ap.add_argument("--template", choices=["v3", "v4"], default="v3",
                    help="v4 checkpoints trained with bos + eos-per-turn need them at inference")
    ap.add_argument("--subset", choices=["grid", "full"], default="grid")
    ap.add_argument("--conditions", default="cur")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seeds", default="1386")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    conds = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    prompts = grid_subset() if args.subset == "grid" else battery()
    # work items: prompt x condition x seed, sharded deterministically
    work = []
    for p in prompts:
        for c in conds:
            for s in seeds:
                work.append((p, c, s))
    work = [w for i, w in enumerate(work) if i % shard_n == shard_i]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                d = json.loads(line)
                done.add((d["id"], d["condition"], d["seed"]))
            except Exception:
                continue
    work = [(p, c, s) for p, c, s in work if (p["id"], c, s) not in done]
    print(f"shard {args.shard}: {len(work)} generations to run "
          f"({len(done)} already done)", flush=True)
    if not work:
        return

    cfg = load_config(str(ROOT / args.config))
    model = Transformer(ModelConfig.from_dict(cfg["model"]))
    sd = torch.load(str(ROOT / args.checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model_state_dict", sd))
    model.eval()
    sp = spm.SentencePieceProcessor()
    sp.load(str(ROOT / "data" / "tokenizer_1.1.model"))
    print("model loaded", flush=True)

    stops = stop_sequences(sp)
    excl = penalty_exclude(sp)

    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for n, (p, cond, seed) in enumerate(work):
            # per-item seed so results are reproducible and shards independent
            torch.manual_seed(seed * 100003 + hash(p["id"]) % 100003)
            kw = dict(CONDITIONS[cond])
            seam = kw.pop("seam", False)
            use_stops = kw.pop("stops", False)
            use_excl = kw.pop("exclude", False)
            if seam:
                v4 = args.template == "v4"
                kw["prompt_ids"] = build_prompt_ids(
                    sp, p["prompt"], eos_between_turns=v4, with_bos=v4)
            if use_stops:
                kw["stop_id_seqs"] = stops
            if use_excl:
                kw["penalty_exclude"] = excl
            t0 = time.time()
            text, meta = generate(
                model, sp, f"User: {p['prompt']}\nAssistant: ",
                max_tokens=args.max_tokens, device=torch.device("cpu"),
                return_new_only=True, return_meta=True, **kw)
            rec = {
                "id": p["id"], "category": p["category"], "condition": cond,
                "seed": seed, "prompt": p["prompt"], "check": p["check"],
                "expect": p["expect"], "difficulty": p["difficulty"],
                "output": text.strip(), "stop": meta["stop"],
                "n_new": meta["n_new"], "secs": round(time.time() - t0, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (n + 1) % 10 == 0:
                rate = (n + 1) / (time.time() - t_start)
                eta = (len(work) - n - 1) / rate / 60
                print(f"  {n+1}/{len(work)} | {rate*60:.1f}/min | eta {eta:.0f} min",
                      flush=True)
    print(f"shard {args.shard} DONE in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
