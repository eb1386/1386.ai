#!/usr/bin/env python3
# global sequence-level shuffle for packed shards.
#
# why: the corpus builder writes sequences in stream order, so when sources
# exhaust at different points every shard ends up with a different domain mix
# (e.g. the tail of the corpus is nearly pure web). training then sees a
# non-stationary distribution and the loss oscillates as the loader walks
# shards. a global shuffle makes every shard match the global mix.
#
# two-pass external shuffle:
#   pass 1 scatter -- stream every input sequence into a random bucket file
#   pass 2 shuffle -- load each bucket, shuffle in RAM, write final shards
# handles optional parallel mask files (train_mask_NNNN.bin) in lockstep.

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def elapsed(t0):
    s = int(time.time() - t0)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def shard_paths(d: Path, split: str):
    files = sorted(p for p in d.glob(f"{split}_*.bin") if "_mask_" not in p.name)
    masks = {}
    for p in d.glob(f"{split}_mask_*.bin"):
        idx = int(p.stem.split("_")[-1])
        masks[idx] = p
    return files, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--buckets", type=int, default=96)
    ap.add_argument("--seqs-per-shard", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--pack-len", type=int, default=1025)
    ap.add_argument("--keep-val", action="store_true", default=True)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    meta_path = in_dir / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    pack_len = int(meta.get("pack_len", args.pack_len))

    files, masks = shard_paths(in_dir, args.split)
    if not files:
        raise SystemExit(f"no {args.split}_*.bin shards in {in_dir}")
    has_masks = len(masks) > 0
    print(f"input: {len(files)} shards, masks={has_masks}, pack_len={pack_len}")

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    bucket_dir = tmp / "_buckets"
    bucket_dir.mkdir()

    rng = np.random.default_rng(args.seed)
    K = args.buckets
    t0 = time.time()

    # pass 1: scatter into buckets
    bfh = [open(bucket_dir / f"b{i:04d}.bin", "wb") for i in range(K)]
    mfh = [open(bucket_dir / f"m{i:04d}.bin", "wb") for i in range(K)] if has_masks else None
    total = 0
    for fi, f in enumerate(files):
        data = np.fromfile(f, dtype=np.uint16)
        n = len(data) // pack_len
        data = data[: n * pack_len].reshape(n, pack_len)
        mdata = None
        if has_masks:
            idx = int(f.stem.split("_")[-1])
            mp = masks.get(idx)
            if mp is None:
                raise SystemExit(f"shard {f.name} has no mask partner")
            mraw = np.fromfile(mp, dtype=np.uint8)
            mdata = mraw[: n * pack_len].reshape(n, pack_len)

        assign = rng.integers(0, K, size=n)
        for b in range(K):
            sel = np.nonzero(assign == b)[0]
            if sel.size:
                data[sel].tofile(bfh[b])
                if has_masks:
                    mdata[sel].tofile(mfh[b])
        total += n
        print(f"  scatter {fi+1}/{len(files)} seqs={total:,} elapsed={elapsed(t0)}", flush=True)

    for fh in bfh:
        fh.close()
    if has_masks:
        for fh in mfh:
            fh.close()

    # pass 2: shuffle each bucket, emit final shards
    out_idx = 0
    carry = np.zeros((0, pack_len), dtype=np.uint16)
    carry_m = np.zeros((0, pack_len), dtype=np.uint8)
    written = 0
    for b in range(K):
        data = np.fromfile(bucket_dir / f"b{b:04d}.bin", dtype=np.uint16)
        n = len(data) // pack_len
        if n == 0:
            continue
        data = data[: n * pack_len].reshape(n, pack_len)
        md = None
        if has_masks:
            mraw = np.fromfile(bucket_dir / f"m{b:04d}.bin", dtype=np.uint8)
            md = mraw[: n * pack_len].reshape(n, pack_len)
        perm = rng.permutation(n)
        data, md = data[perm], (md[perm] if has_masks else None)

        data = np.concatenate([carry, data], axis=0)
        if has_masks:
            md = np.concatenate([carry_m, md], axis=0)
        while len(data) >= args.seqs_per_shard:
            chunk, data = data[: args.seqs_per_shard], data[args.seqs_per_shard:]
            chunk.tofile(tmp / f"{args.split}_{out_idx:04d}.bin")
            if has_masks:
                mchunk, md = md[: args.seqs_per_shard], md[args.seqs_per_shard:]
                mchunk.tofile(tmp / f"{args.split}_mask_{out_idx:04d}.bin")
            written += len(chunk)
            out_idx += 1
        carry = data
        carry_m = md if has_masks else carry_m
        print(f"  shuffle bucket {b+1}/{K} -> {out_idx} shards elapsed={elapsed(t0)}", flush=True)

    if len(carry):
        carry.tofile(tmp / f"{args.split}_{out_idx:04d}.bin")
        if has_masks:
            carry_m.tofile(tmp / f"{args.split}_mask_{out_idx:04d}.bin")
        written += len(carry)
        out_idx += 1

    shutil.rmtree(bucket_dir)

    # carry over val split + meta
    for p in in_dir.glob("val_*.bin"):
        shutil.copy2(p, tmp / p.name)
    meta = dict(meta)
    meta["n_train_sequences"] = int(written)
    meta["n_train_shards"] = int(out_idx)
    meta["globally_shuffled"] = True
    meta["shuffle_seed"] = int(args.seed)
    meta["shuffle_buckets"] = int(K)
    with open(tmp / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=True)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.move(str(tmp), str(out_dir))
    print(f"\nDone in {elapsed(t0)}: {written:,} sequences -> {out_idx} shards in {out_dir}")


if __name__ == "__main__":
    main()
