#!/usr/bin/env python3
# honest, decontaminated head-to-head benchmark for plasma models.
# automated objective scorers, identical for every model:
#   factual   -> keyword match
#   math      -> RANDOM operands, parse final number (uncontaminable)
#   reasoning -> RANDOM word problems, parse number
#   code      -> EXECUTE generated function against hidden tests
#   instruct  -> count distinct listed items >= n
#   essay     -> length + on-topic + non-repetitive + non-refusal
#   convo     -> coherent + non-refusal
# greedy decoding (temperature 0) for determinism. same settings for all models.

import argparse
import ast
import json
import re
import threading
import time
from collections import Counter
from pathlib import Path

import torch
import sentencepiece as spm

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint
from src.inference.generate import generate
from scripts.eval_prompts import FACTUAL, CODE, INSTRUCTION, ESSAY, CONVO

# default models to compare (skipped if checkpoint missing)
DEFAULT_MODELS = [
    ("1.0", "checkpoints/finetune_1.0_final.pt", "configs/finetune_1.0.yaml"),
    ("1.1_old", "checkpoints/finetune_1.1_final.pt", "configs/finetune_1.1.yaml"),
    ("1.1_new", "checkpoints/finetune_1.1_v2_final.pt", "configs/finetune_1.1_v2.yaml"),
]

REFUSALS = ["as an ai language model", "as an ai", "i am an ai", "i'm an ai",
            "as a language model", "i cannot help", "i can't help"]


def load(ckpt, config, device):
    cfg = load_config(config)
    model = Transformer(ModelConfig.from_dict(cfg["model"])).to(device)
    load_checkpoint(ckpt, model)
    model.eval()
    tok = spm.SentencePieceProcessor()
    tok.load(cfg["data"]["tokenizer_path"])
    return model, tok


def gen(model, tok, user, device, max_tokens=160):
    prompt = f"User: {user}\nAssistant:"
    out = generate(model, tok, prompt, max_tokens=max_tokens, temperature=0.0,
                   top_k=1, top_p=1.0, repetition_penalty=1.3, device=device,
                   return_new_only=True)
    out = out.strip()
    for stop in ["\nUser:", "\nAssistant:", "\nQuestion:", "\n\n\n"]:
        if stop in out:
            out = out[: out.index(stop)]
    return out.strip()


def nums(s):
    return re.findall(r"-?\d+", s.replace(",", ""))


def rep_ratio(text, n=3):
    toks = re.findall(r"[a-z]+", text.lower())
    if len(toks) < n + 3:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return Counter(grams).most_common(1)[0][1] / len(grams)


def is_refusal(text):
    low = text.lower()
    return any(r in low for r in REFUSALS)


# ---- scorers (return 1/0) ----
def score_factual(resp, kw):
    low = " " + resp.lower() + " "
    return int(any(k in low for k in kw))


def score_math(resp, expected):
    return int(str(expected) in nums(resp))


def score_instruction(resp, n):
    parts = re.split(r",|\band\b|\n|;|\d+\.|•|\*", resp.lower())
    items = [p.strip() for p in parts if len(p.strip()) >= 2]
    items = list(dict.fromkeys(items))
    return int(len(items) >= n)


def score_essay(resp, kw):
    if len(resp.split()) < 18:
        return 0
    low = resp.lower()
    on_topic = sum(1 for k in kw if k in low) >= 2
    return int(on_topic and rep_ratio(resp) < 0.16 and not is_refusal(resp))


def score_convo(resp):
    if len(resp.split()) < 3:
        return 0
    return int(not is_refusal(resp) and rep_ratio(resp) < 0.22)


def run_code(code, fn, tests, timeout=3.0):
    """exec generated code in restricted ns, run tests in a watchdog thread."""
    # extract code body. an instruction-tuned model typically wraps the answer in
    # chatter ("Sure! Here is ... Hope that helps!"), so strip leading prose at the
    # first def and then drop trailing lines until the block actually parses --
    # otherwise correct code scores 0 purely because of the closing pleasantry.
    m = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
    body = m.group(1) if m else code
    if "def " in body:
        body = body[body.index("def "):]
    lines = body.split("\n")
    while lines:
        try:
            ast.parse("\n".join(lines))
            break
        except SyntaxError:
            lines.pop()
    if not lines:
        return 0
    body = "\n".join(lines)
    safe_builtins = {"range": range, "len": len, "int": int, "str": str, "float": float,
                     "bool": bool, "abs": abs, "min": min, "max": max, "sum": sum,
                     "enumerate": enumerate, "list": list, "dict": dict, "set": set,
                     "sorted": sorted, "reversed": reversed, "True": True, "False": False,
                     "None": None, "ord": ord, "chr": chr, "map": map, "filter": filter,
                     "zip": zip, "round": round, "tuple": tuple}
    ns = {"__builtins__": safe_builtins}
    result = {"ok": 0}

    def worker():
        try:
            exec(body, ns)
            f = ns.get(fn)
            if f is None:
                return
            for args, expected in tests:
                if f(*args) != expected:
                    return
            result["ok"] = 1
        except Exception:
            return

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    return result["ok"]


def make_math(rng):
    items = []
    for _ in range(8):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        items.append((f"What is {a} + {b}?", a + b))
    for _ in range(6):
        a, b = rng.randint(40, 99), rng.randint(11, 39)
        items.append((f"What is {a} - {b}?", a - b))
    for _ in range(6):
        a, b = rng.randint(3, 12), rng.randint(3, 12)
        items.append((f"What is {a} times {b}?", a * b))
    return items


def make_reasoning(rng):
    items = []
    for _ in range(5):
        s, h = rng.randint(20, 80), rng.randint(2, 6)
        items.append((f"If a car travels at {s} miles per hour for {h} hours, how far does it travel?", s * h))
    for _ in range(5):
        a, b = rng.randint(20, 60), rng.randint(5, 19)
        items.append((f"A shop has {a} apples and sells {b}. How many apples are left?", a - b))
    for _ in range(5):
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        items.append((f"There are {a} boxes with {b} balls in each box. How many balls are there in total?", a * b))
    return items


def eval_model(label, ckpt, config, device, seed):
    import random
    rng = random.Random(seed)
    model, tok = load(ckpt, config, device)
    n_params = model.count_parameters()
    cats, samples = {}, {}

    def run(name, prompts, scorer, max_tokens):
        correct = 0
        ex = []
        for item in prompts:
            q = item[0] if isinstance(item, tuple) else item["q"] if isinstance(item, dict) else item
            resp = gen(model, tok, q, device, max_tokens=max_tokens)
            s = scorer(item, resp)
            correct += s
            if len(ex) < 3:
                ex.append({"q": q, "a": resp[:200], "ok": s})
        cats[name] = {"score": correct, "total": len(prompts),
                      "acc": round(correct / len(prompts), 3)}
        samples[name] = ex

    run("factual", FACTUAL, lambda it, r: score_factual(r, it["kw"]), 64)
    run("math", make_math(rng), lambda it, r: score_math(r, it[1]), 48)
    run("reasoning", make_reasoning(rng), lambda it, r: score_math(r, it[1]), 64)
    run("code", CODE, lambda it, r: run_code(r, it["fn"], it["tests"]), 220)
    run("instruction", INSTRUCTION, lambda it, r: score_instruction(r, it["n"]), 64)
    run("essay", ESSAY, lambda it, r: score_essay(r, it["kw"]), 200)
    run("convo", [{"q": q} for q in CONVO], lambda it, r: score_convo(r), 96)

    total = sum(c["score"] for c in cats.values())
    grand = sum(c["total"] for c in cats.values())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"label": label, "params": n_params, "checkpoint": ckpt,
            "categories": cats, "total": total, "grand_total": grand,
            "overall_acc": round(total / grand, 3), "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/benchmark_v2.json")
    ap.add_argument("--seed", type=int, default=1386)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--add", action="append", default=[],
                    help="label:checkpoint:config (repeatable)")
    ap.add_argument("--only", default=None, help="comma list of labels to run")
    args = ap.parse_args()
    device = torch.device(args.device)

    models = list(DEFAULT_MODELS)
    for spec in args.add:
        label, ckpt, config = spec.split(":", 2)
        models.append((label, ckpt, config))
    if args.only:
        keep = set(args.only.split(","))
        models = [m for m in models if m[0] in keep]
    models = [m for m in models if Path(m[1]).exists()]
    if not models:
        raise SystemExit("no model checkpoints found to benchmark")

    results = []
    for label, ckpt, config in models:
        print(f"\n=== evaluating {label} ({ckpt}) ===", flush=True)
        t0 = time.time()
        r = eval_model(label, ckpt, config, device, args.seed)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"  {label}: overall {r['total']}/{r['grand_total']} "
              f"({r['overall_acc']:.1%}) in {r['seconds']}s", flush=True)

    # scorecard
    cat_names = ["factual", "math", "reasoning", "code", "instruction", "essay", "convo"]
    print("\n" + "=" * 72)
    print(f"{'category':<13}" + "".join(f"{r['label']:>14}" for r in results))
    print("-" * 72)
    for c in cat_names:
        row = f"{c:<13}"
        for r in results:
            cell = r["categories"][c]
            row += f"{cell['score']:>5}/{cell['total']:<3}({cell['acc']:.0%})".rjust(14)
        print(row)
    print("-" * 72)
    row = f"{'OVERALL':<13}"
    for r in results:
        row += f"{r['total']:>4}/{r['grand_total']:<3}({r['overall_acc']:.0%})".rjust(14)
    print(row)
    print("=" * 72)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "results": results}, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
