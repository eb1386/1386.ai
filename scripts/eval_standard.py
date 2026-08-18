#!/usr/bin/env python3
# standard multiple-choice benchmarks (lm-eval-harness style) for external
# comparability. these are the benchmarks small-model papers report, so they
# tell us where plasma sits versus SmolLM2-360M / Qwen2.5-0.5B / Pythia-410M
# rather than only versus plasma 1.0.
#
# scoring is log-likelihood ranking over the answer choices -- no generation,
# no decoding settings, no keyword matching. reports both raw accuracy and
# length-normalized accuracy (acc_norm), which is the convention for these sets.

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint

# random-chance baselines, for honesty about what is signal vs noise
BASELINE = {"hellaswag": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25,
            "piqa": 0.50, "openbookqa": 0.25, "winogrande": 0.50, "boolq": 0.50}


def build_tasks(limit):
    """yield (task_name, [ {context, choices, label} ... ])."""
    from datasets import load_dataset
    out = {}

    def take(ds, n):
        return ds.select(range(min(n, len(ds))))

    try:
        ds = take(load_dataset("Rowan/hellaswag", split="validation"), limit)
        out["hellaswag"] = [{
            "context": f"{r['activity_label']}: {r['ctx']}",
            "choices": [" " + e.strip() for e in r["endings"]],
            "label": int(r["label"]),
        } for r in ds if str(r.get("label", "")).isdigit()]
    except Exception as e:
        print(f"  [skip] hellaswag: {type(e).__name__}")

    for cfg, key in (("ARC-Easy", "arc_easy"), ("ARC-Challenge", "arc_challenge")):
        try:
            ds = take(load_dataset("allenai/ai2_arc", cfg, split="validation"), limit)
            rows = []
            for r in ds:
                labels = r["choices"]["label"]
                texts = r["choices"]["text"]
                if r["answerKey"] not in labels:
                    continue
                rows.append({"context": f"Question: {r['question']}\nAnswer:",
                             "choices": [" " + t for t in texts],
                             "label": labels.index(r["answerKey"])})
            out[key] = rows
        except Exception as e:
            print(f"  [skip] {key}: {type(e).__name__}")

    for repo in ("baber/piqa", "ybisk/piqa"):
        try:
            ds = take(load_dataset(repo, split="validation"), limit)
            out["piqa"] = [{"context": f"Question: {r['goal']}\nAnswer:",
                            "choices": [" " + r["sol1"], " " + r["sol2"]],
                            "label": int(r["label"])} for r in ds]
            break
        except Exception as e:
            print(f"  [skip] piqa via {repo}: {type(e).__name__}")

    try:
        ds = take(load_dataset("allenai/openbookqa", "main", split="validation"), limit)
        rows = []
        for r in ds:
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            rows.append({"context": r["question_stem"],
                         "choices": [" " + t for t in r["choices"]["text"]],
                         "label": labels.index(r["answerKey"])})
        out["openbookqa"] = rows
    except Exception as e:
        print(f"  [skip] openbookqa: {type(e).__name__}")

    try:
        ds = take(load_dataset("google/boolq", split="validation"), limit)
        out["boolq"] = [{"context": f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:",
                         "choices": [" no", " yes"],
                         "label": int(bool(r.get("answer", r.get("label", 0))))}
                        for r in ds]
    except Exception as e:
        print(f"  [skip] boolq: {type(e).__name__}")

    return out


@torch.no_grad()
def score_choice(model, tok, context_ids, choice_ids, device, max_len):
    """sum log P(choice tokens | context). returns (sum_logprob, n_tokens, n_chars)."""
    ids = context_ids + choice_ids
    if len(ids) > max_len:
        ids = ids[-max_len:]
        n_choice = min(len(choice_ids), max_len)
    else:
        n_choice = len(choice_ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        logits = model(x)
    logits = logits[0].float()
    logprobs = F.log_softmax(logits, dim=-1)
    total = 0.0
    # predict token t from position t-1
    for i in range(len(ids) - n_choice, len(ids)):
        total += logprobs[i - 1, ids[i]].item()
    return total, n_choice


def eval_model(label, ckpt, config, tasks, device, half=False):
    cfg = load_config(config)
    model = Transformer(ModelConfig.from_dict(cfg["model"])).to(device)
    load_checkpoint(ckpt, model)
    model.eval()
    if half:
        # bf16 weights halve the footprint (~2.1GB -> ~1.05GB) so this can run
        # alongside a live training job without pushing the card over.
        model = model.to(torch.bfloat16)
    tok = spm.SentencePieceProcessor()
    tok.load(cfg["data"]["tokenizer_path"])
    max_len = cfg["model"]["max_seq_len"]

    results = {}
    for task, rows in tasks.items():
        if not rows:
            continue
        n_ok = n_ok_norm = 0
        t0 = time.time()
        for r in rows:
            ctx_ids = tok.encode(r["context"], out_type=int)
            scores, norms = [], []
            for ch in r["choices"]:
                ch_ids = tok.encode(ch, out_type=int)
                if not ch_ids:
                    ch_ids = tok.encode(" " + ch.strip(), out_type=int) or [tok.unk_id()]
                s, n = score_choice(model, tok, ctx_ids, ch_ids, device, max_len)
                scores.append(s)
                norms.append(s / max(1, len(ch)))  # per-character normalization
            if int(max(range(len(scores)), key=lambda i: scores[i])) == r["label"]:
                n_ok += 1
            if int(max(range(len(norms)), key=lambda i: norms[i])) == r["label"]:
                n_ok_norm += 1
        n = len(rows)
        results[task] = {"acc": round(n_ok / n, 4), "acc_norm": round(n_ok_norm / n, 4),
                         "n": n, "seconds": round(time.time() - t0, 1)}
        print(f"    {task:14s} acc {n_ok/n:.3f}  acc_norm {n_ok_norm/n:.3f}  "
              f"(n={n}, baseline {BASELINE.get(task, 0.25):.2f})", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="examples per task")
    ap.add_argument("--out", default="logs/eval_standard.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", action="append", default=[],
                    help="label:checkpoint:config (repeatable)")
    ap.add_argument("--half", action="store_true",
                    help="load weights in bf16 (safe to run beside a training job)")
    args = ap.parse_args()

    models = [m.split(":", 2) for m in args.model] or [
        ["1.0", "checkpoints/finetune_1.0_final.pt", "configs/finetune_1.0.yaml"],
        ["1.1_old", "checkpoints/finetune_1.1_final.pt", "configs/finetune_1.1.yaml"],
    ]
    models = [m for m in models if Path(m[1]).exists()]
    if not models:
        raise SystemExit("no checkpoints found")

    print("loading benchmark datasets...", flush=True)
    tasks = build_tasks(args.limit)
    print(f"tasks: {', '.join(f'{k}({len(v)})' for k, v in tasks.items())}\n", flush=True)

    device = torch.device(args.device)
    all_results = {}
    for label, ckpt, config in models:
        print(f"=== {label} ({ckpt}) ===", flush=True)
        all_results[label] = eval_model(label, ckpt, config, tasks, device, half=args.half)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"limit": args.limit, "baselines": BASELINE, "results": all_results}, indent=2),
        encoding="utf-8")

    # scorecard
    names = list(all_results)
    task_names = sorted({t for r in all_results.values() for t in r})
    print("\n" + "=" * (18 + 16 * len(names)))
    print(f"{'task':<16}{'chance':>7}" + "".join(f"{n:>16}" for n in names))
    print("-" * (18 + 16 * len(names)))
    for t in task_names:
        row = f"{t:<16}{BASELINE.get(t,0.25):>7.2f}"
        for n in names:
            v = all_results[n].get(t)
            row += f"{(v['acc_norm'] if v else float('nan')):>16.3f}"
        print(row)
    print("=" * (18 + 16 * len(names)))
    print("(acc_norm = length-normalized accuracy, the standard reported metric)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
