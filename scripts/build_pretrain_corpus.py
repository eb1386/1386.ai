#!/usr/bin/env python3
# clean plasma 1.1 pretrain corpus
# broad, non-contaminated, single-pass (no upsampling), document-level val split.
# sources: fineweb-edu (raw, light quality filter), wikipedia, code, stackexchange, arxiv.
# NO instruct/synthetic data (kept disjoint from SFT to avoid pretrain/SFT leakage).

import argparse
import html as html_lib
import os
import re
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import sentencepiece as spm
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.quality import filter_and_score

RAW = ROOT / "data" / "raw_1.1"
DEDUPED = ROOT / "data" / "deduped_1.1"
OUT = ROOT / "data" / "shards_1.1_v2"
TOKENIZER = ROOT / "data" / "tokenizer_1.1.model"
SEQ_LEN = 1024
PACK_LEN = SEQ_LEN + 1

# source -> (path, weight). weights are targets; deficit sampling fills from
# whatever is still available as small sources exhaust.
SOURCES = {
    "fineweb_edu":   (RAW / "fineweb_edu_hq.txt",      0.42),
    "wikipedia":     (DEDUPED / "wikipedia_clean.txt", 0.20),
    "code":          (RAW / "code_clean.txt",          0.15),
    "stackexchange": (RAW / "stackexchange_clean.txt", 0.13),
    "arxiv":         (DEDUPED / "arxiv_clean.txt",     0.10),
}

_TAG = re.compile(r"<[^>]+>")


def elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def text_docs(path: Path, min_chars: int = 200):
    """yield blank-line-delimited documents, stripping CRLF cleanly."""
    buffer = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                if buffer:
                    doc = "\n".join(buffer).strip()
                    if len(doc) >= min_chars:
                        yield doc
                    buffer = []
            else:
                buffer.append(line)
        if buffer:
            doc = "\n".join(buffer).strip()
            if len(doc) >= min_chars:
                yield doc


def clean_stackexchange(text: str) -> str:
    """strip html markup, keep q/a text and code."""
    text = _TAG.sub("", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def source_iter(name: str, path: Path):
    """per-source cleaning + filtering. only fineweb gets the prose quality
    classifier (it is web crawl); code/stack/wiki/arxiv would be wrongly
    nuked by prose heuristics (low stopword ratio, html, math symbols)."""
    if name == "fineweb_edu":
        for doc in text_docs(path):
            ok, _ = filter_and_score(doc, min_score=0.45)
            if ok:
                yield doc
    elif name == "stackexchange":
        for doc in text_docs(path, min_chars=120):
            doc = clean_stackexchange(doc)
            if len(doc) >= 120:
                yield doc
    else:  # wikipedia, arxiv, code -- already cleaned/curated upstream
        yield from text_docs(path)


def choose_source(active, weights, source_tokens, total_tokens, rng):
    if total_tokens <= 0:
        return rng.choice(active)
    best, best_deficit = active[0], -10.0
    for name in active:
        actual = source_tokens[name] / max(1, total_tokens)
        deficit = weights[name] - actual
        if deficit > best_deficit:
            best, best_deficit = name, deficit
    return best


def flush(train_seqs, val_seqs, out_dir, rng_np, state):
    if train_seqs:
        idx = rng_np.permutation(len(train_seqs))
        arr = np.stack([train_seqs[i] for i in idx]).astype(np.uint16)
        arr.tofile(out_dir / f"train_{state['train_idx']:04d}.bin")
        state["train_idx"] += 1
        state["train_sequences"] += len(train_seqs)
    if val_seqs:
        idx = rng_np.permutation(len(val_seqs))
        arr = np.stack([val_seqs[i] for i in idx]).astype(np.uint16)
        with open(out_dir / "val_0000.bin", "ab") as f:
            arr.tofile(f)
        state["val_sequences"] += len(val_seqs)


def append_tokens(buffer, seqs, toks):
    buffer.extend(toks)
    while len(buffer) >= PACK_LEN:
        seqs.append(np.asarray(buffer[:PACK_LEN], dtype=np.uint16))
        del buffer[:PACK_LEN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--target-tokens", type=int, default=8_000_000_000)
    ap.add_argument("--min-tokens", type=int, default=3_000_000_000,
                    help="finalize (not fail) if at least this many tokens were built")
    ap.add_argument("--chunk-seqs", type=int, default=100_000)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260607)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and (out_dir / "meta.yaml").exists() and not args.force:
        print(f"[skip] existing completed shards: {out_dir}")
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + f".tmp.{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))
    eos_id = sp.eos_id()

    sources = {k: v[0] for k, v in SOURCES.items() if v[0].exists() and v[0].stat().st_size > 0}
    weights = {k: SOURCES[k][1] for k in sources}
    missing = [k for k in ("code", "stackexchange", "fineweb_edu") if k not in sources]
    if missing:
        raise SystemExit(f"Missing required pretrain sources: {missing}")
    tw = sum(weights.values())
    weights = {k: v / tw for k, v in weights.items()}

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    iters = {name: iter(source_iter(name, p)) for name, p in sources.items()}
    active = list(sources)

    train_buffer, val_buffer = [], []
    train_seqs, val_seqs = [], []
    source_tokens = {n: 0 for n in sources}
    source_docs = {n: 0 for n in sources}
    state = {"train_idx": 0, "train_sequences": 0, "val_sequences": 0}
    total_tokens = docs = errors = 0
    t0 = time.time()

    print("Plasma 1.1 clean pretrain corpus build", flush=True)
    print(f"  target_tokens: {args.target_tokens:,}  out: {out_dir}", flush=True)
    for name in active:
        print(f"    {name:14s} {weights[name]:5.1%} {sources[name]}", flush=True)

    while active and total_tokens < args.target_tokens:
        name = choose_source(active, weights, source_tokens, total_tokens, rng)
        try:
            doc = next(iters[name])
        except StopIteration:
            active.remove(name)
            print(f"  exhausted {name} at {total_tokens/1e9:.3f}B tokens", flush=True)
            continue
        except Exception:
            errors += 1
            continue

        try:
            toks = sp.encode(doc, out_type=int)
        except Exception:
            errors += 1
            continue
        if len(toks) < 32:
            continue
        toks.append(eos_id)

        is_val = rng.random() < args.val_frac
        append_tokens(val_buffer if is_val else train_buffer,
                      val_seqs if is_val else train_seqs, toks)
        source_tokens[name] += len(toks)
        source_docs[name] += 1
        total_tokens += len(toks)
        docs += 1

        if len(train_seqs) >= args.chunk_seqs:
            flush(train_seqs, [], tmp_dir, rng_np, state)
            train_seqs.clear()
        if len(val_seqs) >= args.chunk_seqs:
            flush([], val_seqs, tmp_dir, rng_np, state)
            val_seqs.clear()

        if docs % 100_000 == 0:
            shares = " ".join(f"{n}={source_tokens[n]/max(1,total_tokens):.0%}" for n in sources)
            print(f"  docs={docs:,} tokens={total_tokens/1e9:.3f}B "
                  f"train_seq={state['train_sequences']:,} err={errors} "
                  f"[{shares}] elapsed={elapsed(t0)}", flush=True)

    flush(train_seqs, val_seqs, tmp_dir, rng_np, state)

    if total_tokens < args.min_tokens:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SystemExit(f"Only built {total_tokens:,} tokens (< min {args.min_tokens:,}); aborting")

    meta = {
        "seq_len": SEQ_LEN, "pack_len": PACK_LEN, "dtype": "uint16",
        "tokenizer": TOKENIZER.name, "vocab_size": sp.get_piece_size(),
        "target_tokens": int(args.target_tokens), "total_tokens": int(total_tokens),
        "n_train_sequences": int(state["train_sequences"]),
        "n_val_sequences": int(state["val_sequences"]),
        "n_train_shards": int(state["train_idx"]),
        "source_tokens": {k: int(v) for k, v in source_tokens.items()},
        "source_docs": {k: int(v) for k, v in source_docs.items()},
        "weights": weights,
        "source_paths": {k: str(v) for k, v in sources.items()},
        "doc_errors_skipped": int(errors),
        "old_1_0_sources": False,
        "instruct_in_pretrain": False,
        "code_source_present": "code" in sources,
        "stackexchange_source_present": "stackexchange" in sources,
        "validation_split": "document_level",
    }
    with open(tmp_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=True)
    shutil.move(str(tmp_dir), str(out_dir))
    print("Done.", flush=True)
    print(yaml.safe_dump(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
