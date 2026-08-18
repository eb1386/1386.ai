#!/usr/bin/env python3
# download extra pretrain corpora that the 1.1 mix is missing.
# math is the big gap: plasma scores ~0% on multi-step reasoning and has no
# math corpus at all. also supports topping up fineweb-edu and code.
#
# writes blank-line-delimited plain text, matching data/raw_1.1/*.txt format.

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_1.1"

SOURCES = {
    # high quality math web text (finemath-4plus is the strictest filter)
    "finemath": dict(repo="HuggingFaceTB/finemath", name="finemath-4plus",
                     split="train", field="text", out="finemath_clean.txt"),
    "finemath3": dict(repo="HuggingFaceTB/finemath", name="finemath-3plus",
                      split="train", field="text", out="finemath3_clean.txt"),
    # more educational web text (we exhausted sample-10BT)
    "fineweb100": dict(repo="HuggingFaceFW/fineweb-edu", name="sample-100BT",
                       split="train", field="text", out="fineweb_edu_hq2.txt"),
    # UNFILTERED web crawl -- pop culture, sports, news, forums. the -Edu subset
    # deliberately filters this out, which is why the model thinks Messi plays
    # basketball. this is the entity/pop-knowledge source.
    "fineweb_general": dict(repo="HuggingFaceFW/fineweb", name="sample-10BT",
                            split="train", field="text", out="fineweb_general.txt"),
    # python code
    "code_py": dict(repo="codeparrot/codeparrot-clean", name=None,
                    split="train", field="content", out="code_py2.txt"),
    # synthetic educational textbooks -- the single highest-value source for
    # small models per the SmolLM2 ablations
    "cosmopedia": dict(repo="HuggingFaceTB/smollm-corpus", name="cosmopedia-v2",
                       split="train", field="text", out="cosmopedia_clean.txt"),
    # python educational subset of the stack
    "stack_edu": dict(repo="HuggingFaceTB/smollm-corpus", name="python-edu",
                      split="train", field="text", out="python_edu.txt"),
}


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--target-gb", type=float, default=3.0)
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-docs", type=int, default=0,
                    help="skip N leading docs (to fetch data beyond an earlier download)")
    args = ap.parse_args()

    from datasets import load_dataset

    spec = SOURCES[args.source]
    out_path = Path(args.out) if args.out else RAW / spec["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path} ({human(out_path.stat().st_size)})")
        return

    target = args.target_gb * 1024 ** 3
    print(f"downloading {args.source}: {spec['repo']} "
          f"{spec['name'] or ''} -> {out_path} (target {args.target_gb} GB)", flush=True)

    kwargs = dict(split=spec["split"], streaming=True)
    if spec["name"]:
        kwargs["name"] = spec["name"]
    ds = load_dataset(spec["repo"], **kwargs)

    written = 0
    docs = 0
    skipped = 0
    t0 = time.time()
    tmp = out_path.with_suffix(".partial")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in ds:
            if skipped < args.skip_docs:
                skipped += 1
                continue
            text = (row.get(spec["field"]) or "").strip()
            if len(text) < args.min_chars:
                continue
            f.write(text.replace("\r\n", "\n").replace("\r", "\n"))
            f.write("\n\n")
            written += len(text) + 2
            docs += 1
            if docs % 50_000 == 0:
                rate = written / max(1e-9, time.time() - t0) / 1024 ** 2
                print(f"  {docs:,} docs | {human(written)} | {rate:.1f} MB/s", flush=True)
            if written >= target:
                break
    tmp.rename(out_path)
    print(f"done: {docs:,} docs, {human(written)} in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
