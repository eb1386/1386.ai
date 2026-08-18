#!/usr/bin/env python3
# remove training samples that contain refusal/identity patterns.
# these patterns teach the model to refuse simple prompts ("hi" -> "I am a
# language model and don't have any specific skills"), poisoning instruction
# following. drops ~5-10% of samples but produces a much more useful model.

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

# patterns to match in the LOWERCASED text
REFUSAL_PATTERNS = [
    "i am a language model",
    "i am an ai",
    "i'm an ai",
    "i am a computer program",
    "as an ai",
    "as a language model",
    "i cannot answer",
    "i can't answer",
    "i do not have access",
    "i don't have access",
    "i am not able to provide",
    "i am unable to",
    "as an artificial intelligence",
    "i'm just an ai",
    "i'm a language model",
    "i don't have personal",
    "i do not have personal",
    "as a helpful assistant",
    "i am only a",
    "i'm only a",
]


def filter_file(in_path: Path) -> tuple[int, int]:
    out_path = in_path.with_suffix(".jsonl.filtered")
    kept = 0
    dropped = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                d = json.loads(line)
                t = d.get("text", "")
            except Exception:
                continue
            low = t.lower()
            if any(p in low for p in REFUSAL_PATTERNS):
                dropped += 1
                continue
            fout.write(line)
            kept += 1
    # replace original
    out_path.replace(in_path)
    return kept, dropped


def main():
    raw_dir = ROOT / "data" / "raw_1.1"
    print(f"Filtering refusal patterns from {raw_dir}/instruct_*.jsonl\n")
    total_kept = 0
    total_drop = 0
    for f in sorted(raw_dir.glob("instruct_*.jsonl")):
        kept, dropped = filter_file(f)
        pct = dropped / max(1, kept + dropped) * 100
        print(f"  {f.name:30s}  kept {kept:>7,}  dropped {dropped:>5,}  ({pct:.1f}%)")
        total_kept += kept
        total_drop += dropped
    pct_total = total_drop / max(1, total_kept + total_drop) * 100
    print(f"\n  TOTAL                          kept {total_kept:>7,}  dropped {total_drop:>5,}  ({pct_total:.1f}%)")


if __name__ == "__main__":
    main()
