#!/usr/bin/env python3
# benchmark plasma 1.1 vs 1.0 on a curated battery
# usage: py scripts/benchmark_1.1_vs_1.0.py [--only 1.1|1.0|both]

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


PROMPTS = [
    # (id, prompt, expect-keywords-in-response, category)
    ("math1",   "What is 47 + 89?",                                          ["136"],                         "math"),
    ("math2",   "What is 2 x 2?",                                            ["4"],                           "math"),
    ("math3",   "If a train travels 60 mph for 2.5 hours, how far does it go?", ["150"],                      "math"),
    ("math4",   "What is 12 squared?",                                       ["144"],                         "math"),
    ("math5",   "I have 3 apples. I eat one and give one to my sister. How many do I have left?", ["1"],     "math"),
    ("fact1",   "Who was the first person to walk on the moon?",             ["armstrong", "neil"],           "fact"),
    ("fact2",   "What is the capital of Australia?",                         ["canberra"],                    "fact"),
    ("fact3",   "Who invented the periodic table of elements?",              ["mendeleev"],                   "fact"),
    ("fact4",   "What is the chemical symbol for gold?",                     ["au"],                          "fact"),
    ("fact5",   "When did World War II end?",                                ["1945"],                        "fact"),
    ("fact6",   "What is the largest planet in our solar system?",           ["jupiter"],                     "fact"),
    ("inst1",   "List exactly 5 fruits. Just the list, nothing else.",       ["apple", "banana"],             "format"),
    ("inst2",   "Write a haiku about autumn.",                               [],                              "format"),
    ("inst3",   "Explain photosynthesis in 2 sentences.",                    ["plant", "light", "energy"],    "format"),
    ("conv1",   "hi",                                                        [],                              "conv"),
    ("conv2",   "Hello! How are you?",                                       [],                              "conv"),
    ("conv3",   "Tell me a joke.",                                           [],                              "conv"),
    ("essay1",  "Write a 200-word essay about a farmer named Bob.",          ["bob", "farm"],                 "essay"),
    ("essay2",  "Write a 200-word essay about coding.",                      ["code", "program"],             "essay"),
    ("reason1", "What weighs more: a pound of feathers or a pound of bricks?", ["same", "equal"],             "reason"),
    ("reason2", "Why is the sky blue?",                                      ["scatter", "wavelength", "light", "atmosphere"], "reason"),
    ("expl1",   "Explain how gravity works.",                                ["mass", "force", "attract"],    "expl"),
    ("expl2",   "What is the difference between weather and climate?",       ["weather", "climate"],          "expl"),
    ("creat1",  "Write a short poem about rain.",                            [],                              "creat"),
    ("creat2",  "Continue this story: 'The old man opened the locked drawer for the first time in thirty years.'", [], "creat"),
]


def load(model_id: str):
    """Load the named model + its tokenizer + config."""
    import torch
    import sentencepiece as spm
    from src.model.config import ModelConfig
    from src.model.transformer import Transformer
    from src.train.utils import load_config, load_checkpoint

    if model_id == "1.0":
        cfg_path = ROOT / "configs" / "finetune_1.0.yaml"
        ckpt_path = ROOT / "checkpoints" / "finetune_1.0_final.pt"
        tok_path = ROOT / "data" / "tokenizer_1.0.model"
    elif model_id == "1.1":
        cfg_path = ROOT / "configs" / "finetune_1.1.yaml"
        ckpt_path = ROOT / "checkpoints" / "finetune_1.1_final.pt"
        tok_path = ROOT / "data" / "tokenizer_1.1.model"
    else:
        raise ValueError(model_id)

    cfg = load_config(str(cfg_path))
    mc = ModelConfig.from_dict(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(mc).to(device)
    load_checkpoint(str(ckpt_path), model)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(str(tok_path))
    return model, sp, device


def gen(model, sp, device, prompt, max_tokens=300, temperature=0.3):
    from src.inference.generate import generate
    full_prompt = f"User: {prompt}\nAssistant: "
    out = generate(
        model, sp, full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.3,
        device=device,
    )
    response = out[len(full_prompt):].strip()
    for stop in ["\nUser:", "\nAssistant:", "\nSystem:", "\nHuman:", "\n\n\n"]:
        if stop in response:
            response = response[:response.index(stop)]
    return response.strip()


def score(prompt, response, expect):
    """Heuristic score 0-2 per response.
    0 = empty, off-topic, gibberish, refusal-style ('I am an AI assistant')
    1 = on-topic but flawed (wrong fact, missing detail)
    2 = correct + coherent + actually attempts the task
    """
    if not response or len(response) < 5:
        return 0
    low = response.lower()

    # bad fingerprints — refusal/identity statements / loops
    bad_marks = [
        "i am an ai assistant", "i am not a human", "i don't have opinions",
        "as an ai", "i cannot answer", "as a language model",
    ]
    for b in bad_marks:
        if b in low:
            return 0

    # if any expected keyword is present, mark 2; else if response is meaningful, mark 1
    if expect:
        if any(k.lower() in low for k in expect):
            return 2
        return 1  # on-topic-ish

    # no specific expected keywords (creative / format) — judge length + coherence loosely
    words = response.split()
    if len(words) < 3:
        return 0
    if "assistant:" in low or "user:" in low:
        return 0  # leaked role labels = catastrophic failure
    return 2


def run_battery(model_id: str, out_path: Path):
    model, sp, device = load(model_id)
    results = []
    by_cat_total = {}
    by_cat_score = {}

    print(f"\n{'=' * 70}")
    print(f"  Plasma {model_id}")
    print(f"{'=' * 70}\n")

    for pid, prompt, expect, cat in PROMPTS:
        t0 = time.time()
        try:
            response = gen(model, sp, device, prompt)
        except Exception as e:
            response = f"<<ERROR: {e}>>"
        dt = time.time() - t0
        s = score(prompt, response, expect)
        by_cat_total[cat] = by_cat_total.get(cat, 0) + 2
        by_cat_score[cat] = by_cat_score.get(cat, 0) + s
        results.append({"id": pid, "cat": cat, "prompt": prompt, "response": response, "expect": expect, "score": s, "dt": dt})
        marker = ["X", "~", "OK"][s]
        preview = response.replace("\n", " ")[:120]
        print(f"  [{marker:2s}] {pid:8s} {cat:6s} ({s}/2)  {preview!r}")

    total = sum(by_cat_score.values())
    max_total = sum(by_cat_total.values())
    print(f"\n  Total: {total}/{max_total}")
    print(f"  By category:")
    for cat in by_cat_total:
        print(f"    {cat:8s}  {by_cat_score[cat]:3d}/{by_cat_total[cat]:3d}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_id, "total": total, "max": max_total,
                   "by_cat_total": by_cat_total, "by_cat_score": by_cat_score,
                   "results": results}, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return total, max_total, by_cat_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["1.0", "1.1", "both"], default="both")
    parser.add_argument("--out-dir", default="logs")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    scores = {}
    if args.only in ("1.0", "both"):
        if (ROOT / "checkpoints" / "finetune_1.0_final.pt").exists():
            scores["1.0"] = run_battery("1.0", out_dir / "bench_1.0.json")
        else:
            print("[skip] no 1.0 checkpoint")

    if args.only in ("1.1", "both"):
        if (ROOT / "checkpoints" / "finetune_1.1_final.pt").exists():
            scores["1.1"] = run_battery("1.1", out_dir / "bench_1.1.json")
        else:
            print("[skip] no 1.1 checkpoint (training not done?)")

    if "1.0" in scores and "1.1" in scores:
        s10 = scores["1.0"][0]
        s11 = scores["1.1"][0]
        m = scores["1.0"][1]
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON")
        print(f"{'=' * 70}")
        print(f"  Plasma 1.0:  {s10}/{m}")
        print(f"  Plasma 1.1:  {s11}/{m}")
        print(f"  Delta:      {'+' if s11 >= s10 else ''}{s11 - s10}")
        verdict = "1.1 BEATS 1.0" if s11 > s10 else ("1.1 TIES 1.0" if s11 == s10 else "1.1 LOSES TO 1.0")
        print(f"  Verdict:    {verdict}")


if __name__ == "__main__":
    main()
