#!/usr/bin/env python3
# plasma 1.1 pretrain corpus builder v3.
#
# fixes the two defects found in v2:
#   1. sources exhausted at different points, so the tail of the corpus was
#      nearly pure web -> non-stationary training distribution. v3 re-opens an
#      exhausted source (up to max_epochs) so the mixture stays on target for
#      the WHOLE build.
#   2. no math data at all. finemath + cosmopedia are now first-class sources.
#
# emits packed uint16 sequences plus a document-level val split. run
# scripts/shuffle_shards.py afterwards for a global sequence-level shuffle.

import argparse
import html as html_lib
import json
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
from src.data.safety import is_unsafe

RAW = ROOT / "data" / "raw_1.1"
DEDUPED = ROOT / "data" / "deduped_1.1"
TOKENIZER = ROOT / "data" / "tokenizer_1.1.model"
SEQ_LEN = 1024
PACK_LEN = SEQ_LEN + 1

# name -> (paths, weight, max_epochs, kind)
# kind drives cleaning/filtering. multiple paths are read in order as one stream.
DEFAULT_MIX = {
    "fineweb_edu":     ([RAW / "fineweb_edu_hq.txt", RAW / "fineweb_edu_hq2.txt"], 0.40, 1, "web"),
    # unfiltered web: pop culture, sports, news, people. the -Edu filter strips
    # this, which is why an edu-only model thinks Messi plays basketball.
    "fineweb_general": ([RAW / "fineweb_general.txt"],                            0.00, 1, "webgen"),
    "code":            ([RAW / "code_clean.txt"],                                 0.16, 3, "code"),
    "math":            ([RAW / "finemath_clean.txt"],                             0.10, 2, "plain"),
    "stackexchange":   ([RAW / "stackexchange_clean.txt"],                        0.10, 2, "stack"),
    "cosmopedia":      ([RAW / "cosmopedia_clean.txt"],                           0.08, 2, "plain"),
    "wikipedia":       ([DEDUPED / "wikipedia_clean.txt"],                        0.10, 2, "plain"),
    "arxiv":           ([DEDUPED / "arxiv_clean.txt"],                            0.06, 2, "plain"),
}

_TAG = re.compile(r"<[^>]+>")


def elapsed(t0):
    s = int(time.time() - t0)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def text_docs(path: Path, min_chars: int = 200):
    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                if buf:
                    doc = "\n".join(buf).strip()
                    if len(doc) >= min_chars:
                        yield doc
                    buf = []
            else:
                buf.append(line)
        if buf:
            doc = "\n".join(buf).strip()
            if len(doc) >= min_chars:
                yield doc


def clean_stack(text: str) -> str:
    text = _TAG.sub("", text)
    text = html_lib.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def one_pass(paths, kind: str):
    """single pass over all files for a source, cleaned/filtered by kind."""
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        if kind == "web":
            for doc in text_docs(path):
                ok, _ = filter_and_score(doc, min_score=0.45)
                if ok:
                    yield doc
        elif kind == "webgen":
            # unfiltered crawl: keep the quality floor LOW so entity/pop-culture
            # content survives (that is the entire point of this source), while
            # still dropping obvious junk pages -- and screen it for slurs,
            # explicit content and spam, which the -Edu filter used to handle
            # for us implicitly.
            for doc in text_docs(path, min_chars=300):
                if is_unsafe(doc):
                    continue
                ok, _ = filter_and_score(doc, min_score=0.30)
                if ok:
                    yield doc
        elif kind == "stack":
            for doc in text_docs(path, min_chars=120):
                doc = clean_stack(doc)
                if len(doc) >= 120:
                    yield doc
        else:
            yield from text_docs(path)


def epoch_iter(name, paths, kind, max_epochs, counter):
    """re-open the source when exhausted, up to max_epochs, so the mixture
    stays on target instead of the source silently dropping out."""
    for ep in range(max_epochs):
        n = 0
        for doc in one_pass(paths, kind):
            n += 1
            yield doc
        counter[name] = ep + 1
        if n == 0:
            return
        print(f"  [{name}] finished epoch {ep+1}/{max_epochs} ({n:,} docs)", flush=True)


def choose(active, weights, tok, total):
    if total <= 0:
        return active[0]
    best, best_def = active[0], -10.0
    for n in active:
        d = weights[n] - tok[n] / max(1, total)
        if d > best_def:
            best, best_def = n, d
    return best


def flush(train_seqs, val_seqs, out_dir, rng, state):
    if train_seqs:
        idx = rng.permutation(len(train_seqs))
        np.stack([train_seqs[i] for i in idx]).astype(np.uint16).tofile(
            out_dir / f"train_{state['train_idx']:04d}.bin")
        state["train_idx"] += 1
        state["train_sequences"] += len(train_seqs)
    if val_seqs:
        idx = rng.permutation(len(val_seqs))
        arr = np.stack([val_seqs[i] for i in idx]).astype(np.uint16)
        with open(out_dir / "val_0000.bin", "ab") as f:
            arr.tofile(f)
        state["val_sequences"] += len(val_seqs)


def append_tokens(buf, seqs, toks):
    buf.extend(toks)
    while len(buf) >= PACK_LEN:
        seqs.append(np.asarray(buf[:PACK_LEN], dtype=np.uint16))
        del buf[:PACK_LEN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "shards_1.1_v3"))
    ap.add_argument("--target-tokens", type=int, default=16_000_000_000)
    ap.add_argument("--min-tokens", type=int, default=5_000_000_000)
    ap.add_argument("--chunk-seqs", type=int, default=100_000)
    ap.add_argument("--val-frac", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--mix", default=None, help="yaml file overriding the mixture weights")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and (out_dir / "meta.yaml").exists() and not args.force:
        print(f"[skip] existing shards: {out_dir}")
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp = out_dir.with_name(out_dir.name + f".tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    mix = {k: dict(paths=v[0], weight=v[1], max_epochs=v[2], kind=v[3])
           for k, v in DEFAULT_MIX.items()}
    if args.mix:
        override = yaml.safe_load(Path(args.mix).read_text(encoding="utf-8"))
        for k, v in (override.get("weights") or {}).items():
            if k in mix:
                mix[k]["weight"] = float(v)
        for k, v in (override.get("max_epochs") or {}).items():
            if k in mix:
                mix[k]["max_epochs"] = int(v)

    # drop zero-weight sources (an anneal mix zeroes some out) and any source
    # with no data on disk, then renormalize
    for k in list(mix):
        if mix[k]["weight"] <= 0:
            print(f"  [drop] {k}: weight 0")
            del mix[k]
        elif not any(p.exists() and p.stat().st_size > 0 for p in mix[k]["paths"]):
            print(f"  [warn] no data for {k}, dropping")
            del mix[k]
    if not mix:
        raise SystemExit("no sources left after filtering")
    tot_w = sum(m["weight"] for m in mix.values())
    for m in mix.values():
        m["weight"] /= tot_w

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))
    eos = sp.eos_id()
    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    epochs_done = {k: 0 for k in mix}
    iters = {k: epoch_iter(k, m["paths"], m["kind"], m["max_epochs"], epochs_done)
             for k, m in mix.items()}
    weights = {k: m["weight"] for k, m in mix.items()}
    active = list(mix)

    train_buf, val_buf, train_seqs, val_seqs = [], [], [], []
    src_tok = {k: 0 for k in mix}
    src_doc = {k: 0 for k in mix}
    state = {"train_idx": 0, "train_sequences": 0, "val_sequences": 0}
    total = docs = errors = 0
    t0 = time.time()

    print("Plasma 1.1 corpus v3")
    print(f"  target {args.target_tokens/1e9:.1f}B tokens -> {out_dir}")
    for k, m in mix.items():
        print(f"    {k:14s} {m['weight']:5.1%} max_epochs={m['max_epochs']} "
              f"{[p.name for p in m['paths']]}")

    while active and total < args.target_tokens:
        name = choose(active, weights, src_tok, total)
        try:
            doc = next(iters[name])
        except StopIteration:
            active.remove(name)
            print(f"  exhausted {name} after {epochs_done[name]} epoch(s) "
                  f"at {total/1e9:.2f}B", flush=True)
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
        toks.append(eos)

        is_val = rng.random() < args.val_frac
        append_tokens(val_buf if is_val else train_buf,
                      val_seqs if is_val else train_seqs, toks)
        src_tok[name] += len(toks)
        src_doc[name] += 1
        total += len(toks)
        docs += 1

        if len(train_seqs) >= args.chunk_seqs:
            flush(train_seqs, [], tmp, rng_np, state)
            train_seqs.clear()
        if len(val_seqs) >= args.chunk_seqs:
            flush([], val_seqs, tmp, rng_np, state)
            val_seqs.clear()

        if docs % 200_000 == 0:
            sh = " ".join(f"{n}={src_tok[n]/max(1,total):.0%}" for n in mix)
            print(f"  docs={docs:,} tok={total/1e9:.3f}B seq={state['train_sequences']:,} "
                  f"[{sh}] {elapsed(t0)}", flush=True)

    flush(train_seqs, val_seqs, tmp, rng_np, state)
    if total < args.min_tokens:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"only {total:,} tokens (< min {args.min_tokens:,})")

    meta = {
        "seq_len": SEQ_LEN, "pack_len": PACK_LEN, "dtype": "uint16",
        "tokenizer": TOKENIZER.name, "vocab_size": sp.get_piece_size(),
        "target_tokens": int(args.target_tokens), "total_tokens": int(total),
        "n_train_sequences": int(state["train_sequences"]),
        "n_val_sequences": int(state["val_sequences"]),
        "n_train_shards": int(state["train_idx"]),
        "source_tokens": {k: int(v) for k, v in src_tok.items()},
        "source_docs": {k: int(v) for k, v in src_doc.items()},
        "source_epochs": dict(epochs_done),
        "weights_target": weights,
        "weights_actual": {k: round(src_tok[k] / max(1, total), 4) for k in src_tok},
        "doc_errors_skipped": int(errors),
        "old_1_0_sources": False, "instruct_in_pretrain": False,
        "validation_split": "document_level",
        "globally_shuffled": False,
    }
    (tmp / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=True), encoding="utf-8")
    shutil.move(str(tmp), str(out_dir))
    print("\nDone.", yaml.safe_dump(meta, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
