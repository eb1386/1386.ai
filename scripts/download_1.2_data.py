#!/usr/bin/env python3
# plasma 1.2 pretrain downloads (~73 GB new raw -> 30B-token corpus)
#
# writes to data/raw_1.2/ with an explicit <|doc|> sentinel line between
# documents. the 1.1 builder split on blank lines, which fragmented source
# files and textbook chapters into ~300-char shreds and silently dropped 19%
# of code; the sentinel keeps documents whole for the 2048-token context.
#
# resumable: each source skips itself if its output exists and is complete.
#
#   python scripts/download_1.2_data.py                 # everything
#   python scripts/download_1.2_data.py --only code     # one source
#   python scripts/download_1.2_data.py --dry-run       # show the plan

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_1.2"
RAW11 = ROOT / "data" / "raw_1.1"
SENTINEL = "<|doc|>"

# (key, hf repo, config, split, field, target_gb, note)
PLAN = [
    # starcoderdata is GATED (needs hf auth) — likely why 1.1's code file
    # ended up empty. codeparrot-clean is ungated, deduplicated python.
    ("code", "codeparrot/codeparrot-clean", None, "train", "content", 12.0,
     "deduplicated python source files, whole-file docs"),
    ("books", "manu/project_gutenberg", None, "en", "text", 14.0,
     "long-form books; 1.1 had zero book data"),
    ("wiki_full", "wikimedia/wikipedia", "20231101.en", "train", "text", 20.0,
     "FULL english wikipedia (1.1 had ~30% of it)"),
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-100BT", "train", "text", 14.0,
     "fresh educational web, offset past the 1.1 pull"),
    ("finemath", "HuggingFaceTB/finemath", "finemath-3plus", "train", "text", 6.0,
     "superset of 4plus on disk; deduped against it at download time"),
    ("cosmopedia", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text", 7.0,
     "synthetic textbooks, offset past the 1.1 pull"),
]

# skip this many leading docs for sources 1.1 already drank from, so the new
# pull is fresh data rather than a byte-identical re-download
SKIP_DOCS = {"fineweb_edu": 5_500_000, "cosmopedia": 3_000_000}


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} TB"


def dedup_hashes_for(key):
    """finemath-3plus contains all of 4plus: hash what we already have"""
    if key != "finemath":
        return None
    prior = RAW11 / "finemath_clean.txt"
    if not prior.exists():
        return set()
    print("  hashing existing finemath-4plus for dedup...", flush=True)
    hashes, buf = set(), []
    with open(prior, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if buf:
                    hashes.add(hashlib.md5("".join(buf)[:2000].encode()).digest())
                    buf = []
            else:
                buf.append(line)
    if buf:
        hashes.add(hashlib.md5("".join(buf)[:2000].encode()).digest())
    print(f"  {len(hashes):,} prior docs hashed", flush=True)
    return hashes


def fetch(key, repo, config, split, field, target_gb, note):
    out = RAW / f"{key}.txt"
    done = RAW / f"{key}.done"
    if done.exists():
        print(f"[skip] {key}: complete ({human(out.stat().st_size)})")
        return
    from datasets import load_dataset

    target = target_gb * 1024 ** 3
    print(f"\n[{key}] {repo} {config or ''} -> {out} (target {target_gb} GB)")
    print(f"        {note}", flush=True)

    kwargs = dict(split=split, streaming=True)
    if config:
        kwargs["name"] = config
    ds = load_dataset(repo, **kwargs)

    prior = dedup_hashes_for(key)
    skip = SKIP_DOCS.get(key, 0)
    written = out.stat().st_size if out.exists() else 0
    mode = "a" if written else "w"
    docs = skipped = deduped = 0
    t0 = time.time()
    with open(out, mode, encoding="utf-8") as f:
        for row in ds:
            if skipped < skip:
                skipped += 1
                continue
            text = (row.get(field) or "").strip()
            if len(text) < 200:
                continue
            if prior is not None:
                h = hashlib.md5(text[:2000].encode()).digest()
                if h in prior:
                    deduped += 1
                    continue
            f.write(text.replace("\r\n", "\n").replace("\r", "\n"))
            f.write(f"\n{SENTINEL}\n")
            written += len(text) + len(SENTINEL) + 2
            docs += 1
            if docs % 50_000 == 0:
                mb_s = written / max(1e-9, time.time() - t0) / 1024 ** 2
                print(f"  {docs:,} docs | {human(written)} | {mb_s:.1f} MB/s"
                      + (f" | deduped {deduped:,}" if deduped else ""), flush=True)
            if written >= target:
                break
    done.write_text(f"{docs} docs, {written} bytes\n")
    print(f"  done: {docs:,} docs, {human(written)}"
          + (f", {deduped:,} deduped" if deduped else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    total = 0
    for key, repo, config, split, field, gb, note in PLAN:
        if args.only and key != args.only:
            continue
        total += gb
        if args.dry_run:
            state = "done" if (RAW / f"{key}.done").exists() else "pending"
            print(f"  [{state:>7}] {key:<12} {gb:5.1f} GB  {repo} {config or ''}")
            continue
        try:
            fetch(key, repo, config, split, field, gb, note)
        except Exception as e:
            print(f"  [ERROR] {key}: {type(e).__name__}: {e}", flush=True)
            print("  continuing with remaining sources; re-run to retry this one")
    print(f"\nplanned total: ~{total:.0f} GB raw "
          f"(disk free is ample; shards add ~120 GB during the build)")


if __name__ == "__main__":
    main()
