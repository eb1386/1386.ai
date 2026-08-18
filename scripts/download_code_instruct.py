#!/usr/bin/env python3
# download code-instruction data for sft v4
#
# the v3 sft set had ~0.8% conversations containing a python function, which
# is why the model cannot write add_numbers(a, b). two small, clean sources:
#   codealpaca-20k  -> data/raw_1.1/instruct_codealpaca.jsonl
#   evol-codealpaca -> data/raw_1.1/instruct_evolcode.jsonl  (python-leaning slice)
# same {"text": "User: ...\nAssistant: ..."} format as the other sources.

import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw_1.1"


def fmt(user, assistant):
    return f"User: {user.strip()}\nAssistant: {assistant.strip()}"


def fence_if_bare(answer):
    """code answers must model good formatting: fence bare code"""
    a = answer.strip()
    if "```" in a:
        return a
    looks_code = bool(re.match(r"^(def |class |import |from |for |while |print\(|#)", a))
    if looks_code:
        return "```python\n" + a + "\n```"
    return a


def download_codealpaca(load_dataset):
    out = OUT_DIR / "instruct_codealpaca.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    print("downloading sahil2801/CodeAlpaca-20k ...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            instr = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            outp = (row.get("output") or "").strip()
            if not instr or not outp:
                continue
            user = instr + ("\n" + inp if inp else "")
            f.write(json.dumps({"text": fmt(user, fence_if_bare(outp))},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"  wrote {n:,} -> {out.name}")


def download_evolcode(load_dataset, target=45000):
    out = OUT_DIR / "instruct_evolcode.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    print("downloading theblackcat102/evol-codealpaca-v1 (python-leaning) ...")
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train",
                      streaming=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            instr = (row.get("instruction") or "").strip()
            outp = (row.get("output") or "").strip()
            if not instr or not outp:
                continue
            blob = (instr + " " + outp).lower()
            # keep python and language-neutral; skip the heavy js/java/c# tail
            if any(k in blob for k in ("javascript", "typescript", " java ",
                                       "c#", "swift", "kotlin", "php", "ruby on")):
                continue
            f.write(json.dumps({"text": fmt(instr, outp)}, ensure_ascii=False) + "\n")
            n += 1
            if n >= target:
                break
    print(f"  wrote {n:,} -> {out.name}")


def main():
    from datasets import load_dataset
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download_codealpaca(load_dataset)
    download_evolcode(load_dataset)


if __name__ == "__main__":
    main()
