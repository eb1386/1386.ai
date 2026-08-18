#!/usr/bin/env python3
# Build corrected Plasma 1.1 pretrain shards without old 1.0 upsampling.

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.quality import filter_and_score


RAW = ROOT / "data" / "raw_1.1"
DEDUPED = ROOT / "data" / "deduped_1.1"
OUT = ROOT / "data" / "shards_1.1_repretrain"
TOKENIZER = ROOT / "data" / "tokenizer_1.1.model"
SEQ_LEN = 1024
PACK_LEN = SEQ_LEN + 1


def elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def text_docs(path: Path, min_chars: int = 200):
    buffer = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
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


def jsonl_text_docs(path: Path, min_chars: int = 80):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "").strip()
            except Exception:
                continue
            if len(text) >= min_chars:
                yield text


def source_iter(name: str, path: Path):
    if name.startswith("instruct_") or name == "synthetic_instruct":
        yield from jsonl_text_docs(path)
        return
    for doc in text_docs(path):
        if name == "fineweb_edu":
            ok, score = filter_and_score(doc, min_score=0.58)
            if not ok or score < 0.58:
                continue
        yield doc


def choose_source(
    rng: random.Random,
    active: list[str],
    weights: dict[str, float],
    source_tokens: dict[str, int],
    total_tokens: int,
) -> str:
    if total_tokens <= 0:
        return rng.choice(active)
    best = active[0]
    best_deficit = -10.0
    for name in active:
        actual = source_tokens[name] / max(1, total_tokens)
        deficit = weights[name] - actual
        if deficit > best_deficit:
            best = name
            best_deficit = deficit
    return best


def flush_sequences(
    train_seqs: list[np.ndarray],
    val_seqs: list[np.ndarray],
    out_dir: Path,
    rng: np.random.Generator,
    shard_state: dict,
):
    if not train_seqs and not val_seqs:
        return
    if train_seqs:
        idx = rng.permutation(len(train_seqs))
        arr = np.stack([train_seqs[i] for i in idx]).astype(np.uint16)
        path = out_dir / f"train_{shard_state['train_idx']:04d}.bin"
        arr.tofile(path)
        shard_state["train_idx"] += 1
        shard_state["train_sequences"] += len(train_seqs)
    if val_seqs:
        idx = rng.permutation(len(val_seqs))
        arr = np.stack([val_seqs[i] for i in idx]).astype(np.uint16)
        with open(out_dir / "val_0000.bin", "ab") as f:
            arr.tofile(f)
        shard_state["val_sequences"] += len(val_seqs)


def append_tokens(buffer: list[int], seqs: list[np.ndarray], toks: list[int]):
    buffer.extend(toks)
    while len(buffer) >= PACK_LEN:
        seq = np.asarray(buffer[:PACK_LEN], dtype=np.uint16)
        del buffer[:PACK_LEN]
        seqs.append(seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--target-tokens", type=int, default=10_500_000_000)
    parser.add_argument("--chunk-seqs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-code-stack", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and (out_dir / "meta.yaml").exists() and not args.force:
        print(f"[skip] existing completed shards: {out_dir}")
        return
    if out_dir.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite incomplete existing directory without --force: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + f".tmp.{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))
    eos_id = sp.eos_id()

    sources = {
        "wikipedia": DEDUPED / "wikipedia_clean.txt",
        "fineweb_edu": RAW / "fineweb_edu_hq.txt",
        "arxiv": DEDUPED / "arxiv_clean.txt",
        "stackexchange": RAW / "stackexchange_clean.txt",
        "code": RAW / "code_clean.txt",
        "instruct_slimorca": RAW / "instruct_slimorca.jsonl",
        "instruct_ultrachat": RAW / "instruct_ultrachat.jsonl",
        "instruct_metamath": RAW / "instruct_metamath.jsonl",
        "synthetic_instruct": RAW / "synthetic_instruct.jsonl",
    }
    weights = {
        "fineweb_edu": 0.30,
        "wikipedia": 0.25,
        "code": 0.16,
        "stackexchange": 0.10,
        "arxiv": 0.09,
        "instruct_metamath": 0.04,
        "instruct_slimorca": 0.025,
        "instruct_ultrachat": 0.025,
        "synthetic_instruct": 0.01,
    }
    sources = {k: v for k, v in sources.items() if v.exists() and v.stat().st_size > 0}
    if args.require_code_stack:
        missing = [name for name in ("code", "stackexchange") if name not in sources]
        if missing:
            raise SystemExit(
                "Missing required broad-domain pretrain sources: "
                + ", ".join(missing)
                + ". Run with --download-missing first or provide raw_1.1 files."
            )
    weights = {k: weights[k] for k in sources}
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    iters = {name: iter(source_iter(name, path)) for name, path in sources.items()}
    active = list(sources)

    train_buffer: list[int] = []
    val_buffer: list[int] = []
    train_seqs: list[np.ndarray] = []
    val_seqs: list[np.ndarray] = []
    source_tokens = {name: 0 for name in sources}
    source_docs = {name: 0 for name in sources}
    shard_state = {"train_idx": 0, "train_sequences": 0, "val_sequences": 0}
    total_tokens = 0
    docs = 0
    t0 = time.time()

    print("Corrected pretrain shard build")
    print(f"  target_tokens: {args.target_tokens:,}")
    print("  sources:")
    for name in active:
        print(f"    {name:20s} {weights[name]:5.1%} {sources[name]}")

    while active and total_tokens < args.target_tokens:
        name = choose_source(rng, active, weights, source_tokens, total_tokens)
        try:
            doc = next(iters[name])
        except StopIteration:
            active.remove(name)
            print(f"  exhausted {name}")
            continue

        toks = sp.encode(doc, out_type=int)
        if len(toks) < 32:
            continue
        toks.append(eos_id)
        is_val_doc = rng.random() < 0.02
        append_tokens(val_buffer if is_val_doc else train_buffer, val_seqs if is_val_doc else train_seqs, toks)
        source_tokens[name] += len(toks)
        source_docs[name] += 1
        total_tokens += len(toks)
        docs += 1

        if len(train_seqs) >= args.chunk_seqs:
            flush_sequences(train_seqs, [], tmp_dir, rng_np, shard_state)
            train_seqs.clear()
        if len(val_seqs) >= args.chunk_seqs:
            flush_sequences([], val_seqs, tmp_dir, rng_np, shard_state)
            val_seqs.clear()

        if docs % 50_000 == 0:
            print(
                f"  docs={docs:,} tokens={total_tokens/1e9:.3f}B "
                f"train_seq={shard_state['train_sequences']:,} "
                f"elapsed={elapsed(t0)}"
            )

    flush_sequences(train_seqs, val_seqs, tmp_dir, rng_np, shard_state)
    if train_buffer:
        train_buffer.clear()
    if val_buffer:
        val_buffer.clear()
    if total_tokens < args.target_tokens:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SystemExit(
            f"Only built {total_tokens:,} tokens before sources exhausted; target was {args.target_tokens:,}"
        )

    meta = {
        "seq_len": SEQ_LEN,
        "pack_len": PACK_LEN,
        "dtype": "uint16",
        "tokenizer": TOKENIZER.name,
        "vocab_size": sp.get_piece_size(),
        "target_tokens": int(args.target_tokens),
        "total_tokens": int(total_tokens),
        "n_train_sequences": int(shard_state["train_sequences"]),
        "n_val_sequences": int(shard_state["val_sequences"]),
        "n_train_shards": int(shard_state["train_idx"]),
        "source_tokens": {k: int(v) for k, v in source_tokens.items()},
        "source_docs": {k: int(v) for k, v in source_docs.items()},
        "weights": weights,
        "source_paths": {k: str(v) for k, v in sources.items()},
        "old_1_0_sources": False,
        "code_source_present": "code" in sources,
        "stackexchange_source_present": "stackexchange" in sources,
        "validation_split": "document_level",
    }
    with open(tmp_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=True)
    shutil.move(str(tmp_dir), str(out_dir))
    print("Done.")
    print(yaml.safe_dump(meta, sort_keys=True))


if __name__ == "__main__":
    main()
