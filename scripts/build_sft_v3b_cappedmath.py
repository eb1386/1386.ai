#!/usr/bin/env python3
# plasma 1.1 SFT builder v3.
#
# improvements over v2:
#   1. WHOLE-CONVERSATION packing. v2 concatenated every conversation into one
#      stream and cut at 1025 tokens, so a sequence could contain the tail of one
#      conversation and the head of another, and attention ran across the seam.
#      v3 bin-packs complete conversations and pads the remainder (loss-masked),
#      so no example ever attends to an unrelated one.
#   2. Conversations longer than the context are dropped rather than truncated
#      mid-answer (a truncated answer teaches the model not to emit EOS).
#   3. Padding is masked out of the loss and the pad ratio is reported.
#   4. Same decontamination against the benchmark prompts as v2.

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_prompts import all_prompt_strings, norm_prompt

RAW = ROOT / "data" / "raw_1.1"
OUT = ROOT / "data" / "sft_shards_1.1_v3b"
TOKENIZER = ROOT / "data" / "tokenizer_1.1.model"
SEQ_LEN = 1024
PACK_LEN = SEQ_LEN + 1

SOURCES = {
    "instruct_slimorca":  120000,
    "instruct_ultrachat":  80000,
    "instruct_wizardlm":   50000,
    "instruct_metamath":   18000,   # capped: Step-N CoT was bleeding into general QA
    "instruct_alpaca":     40000,
    "instruct_hh_helpful": 30000,
    "instruct_oasst":      25000,
    "instruct_capybara":   15000,
    "instruct_dolly":      15000,
    "instruct_no_robots":  10000,
}

ROLE_RE = re.compile(r"(?:^|\n)(User|Assistant): ?")
BAD_PATTERNS = (
    "as an ai language model", "as an ai", "i am an ai", "i'm an ai",
    "as a language model", "i am a language model",
    "i don't have access", "i do not have access",
    "i cannot answer", "i can't answer", "i cannot provide", "i'm just an ai",
)


def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def bad_sample(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in BAD_PATTERNS) or len(text) < 24


def first_user(text: str) -> str:
    m = re.search(r"(?:^|\n)User: ?(.*?)(?:\nAssistant:|$)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def build_mask(text: str, sp, max_len=None):
    """tokens + mask; 1 only on assistant content (and its EOS).

    if max_len is given and the conversation is longer, truncate at a TURN
    boundary so the sequence still ends on a complete assistant answer. that
    keeps long multi-turn data instead of discarding it, without ever teaching
    the model to stop mid-sentence.
    """
    bounds = []
    for m in ROLE_RE.finditer(text):
        role = "user" if m.group(1) == "User" else "assistant"
        bounds.append((role, m.start(1), m.end()))
    if not bounds:
        return None, None
    tokens, bits = [], []
    if bounds[0][1] > 0:
        p = sp.encode(text[: bounds[0][1]], out_type=int)
        tokens += p
        bits += [0] * len(p)

    end_of_last_assistant = None       # token count after each completed assistant turn
    for i, (role, rs, le) in enumerate(bounds):
        nxt = bounds[i + 1][1] if i + 1 < len(bounds) else len(text)
        lab = sp.encode(text[rs:le], out_type=int)
        tokens += lab
        bits += [0] * len(lab)
        cont = sp.encode(text[le:nxt], out_type=int)
        tokens += cont
        bits += [1 if role == "assistant" else 0] * len(cont)
        if role == "assistant" and (max_len is None or len(tokens) <= max_len):
            end_of_last_assistant = len(tokens)

    if max_len is not None and len(tokens) > max_len:
        if not end_of_last_assistant:
            return None, None          # not even one full answer fits
        tokens = tokens[:end_of_last_assistant]
        bits = bits[:end_of_last_assistant]

    if len(tokens) < 8 or not any(bits):
        return None, None
    return tokens, bits


def load_source(name, cap, rng):
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        print(f"  [warn] missing {path}")
        return []
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if t:
                rows.append(t)
    rng.shuffle(rows)
    return rows[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and (out_dir / "meta.yaml").exists() and not args.force:
        print(f"[skip] existing: {out_dir}")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))
    eos, pad = sp.eos_id(), sp.pad_id()
    if pad is None or pad < 0:
        pad = 0
    rng = random.Random(args.seed)
    exclude = all_prompt_strings()
    t0 = time.time()

    records = []
    for name, cap in SOURCES.items():
        rows = load_source(name, cap, rng)
        records += [(t, name) for t in rows]
        print(f"  loaded {name}: {len(rows):,}", flush=True)
    rng.shuffle(records)

    # tokenize into complete conversations
    convs = []
    src_counts, skips = {}, {}
    dropped_long = decontam = 0
    for text, source in records:
        text = normalize_text(text)
        if bad_sample(text):
            skips[source] = skips.get(source, 0) + 1
            continue
        if norm_prompt(first_user(text)) in exclude:
            decontam += 1
            continue
        # truncate at a turn boundary rather than discarding long conversations
        toks, bits = build_mask(text, sp, max_len=PACK_LEN - 1)
        if toks is None:
            skips[source] = skips.get(source, 0) + 1
            continue
        toks.append(eos)
        bits.append(1)
        if len(toks) > PACK_LEN:
            dropped_long += 1
            continue
        convs.append((toks, bits, source))
        src_counts[source] = src_counts.get(source, 0) + 1

    if not convs:
        raise SystemExit("no conversations survived filtering")
    print(f"\n  usable conversations: {len(convs):,} "
          f"(dropped_too_long={dropped_long:,}, decontam={decontam})", flush=True)

    # best-fit-decreasing bin packing into PACK_LEN bins, whole conversations only.
    # open bins are bucketed by remaining capacity (32-token granularity) so we can
    # find the tightest bin that fits in O(1) instead of scanning a truncated window
    # (the earlier windowed scan threw away partly-filled bins -> 21.7% padding).
    GRAN = 32
    NB = PACK_LEN // GRAN + 1
    order = sorted(range(len(convs)), key=lambda i: -len(convs[i][0]))
    bins = []                       # list of [tokens, bits]
    buckets = [[] for _ in range(NB)]   # capacity bucket -> [bin indices]
    for i in order:
        toks, bits, _ = convs[i]
        need = len(toks)
        b0 = (need + GRAN - 1) // GRAN      # smallest bucket that can hold `need`
        idx = None
        for b in range(b0, NB):
            if buckets[b]:
                idx = buckets[b].pop()
                break
        if idx is None:
            bins.append([list(toks), list(bits)])
            idx, rem = len(bins) - 1, PACK_LEN - need
        else:
            bt, bb = bins[idx]
            bt.extend(toks)
            bb.extend(bits)
            rem = PACK_LEN - len(bt)
        if rem >= GRAN:
            buckets[rem // GRAN].append(idx)

    n_seq = len(bins)
    packed = np.full((n_seq, PACK_LEN), pad, dtype=np.uint16)
    masks = np.zeros((n_seq, PACK_LEN), dtype=np.uint8)
    real_tokens = 0
    for i, (bt, bb) in enumerate(bins):
        L = min(len(bt), PACK_LEN)
        packed[i, :L] = np.asarray(bt[:L], dtype=np.uint16)
        masks[i, :L] = np.asarray(bb[:L], dtype=np.uint8)
        real_tokens += L

    rng_np = np.random.default_rng(args.seed)
    perm = rng_np.permutation(n_seq)
    packed, masks = packed[perm], masks[perm]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = max(1, int(n_seq * args.val_frac))
    n_train = n_seq - n_val
    per_shard = 100_000
    idx = 0
    for start in range(0, n_train, per_shard):
        end = min(start + per_shard, n_train)
        packed[start:end].tofile(out_dir / f"train_{idx:04d}.bin")
        masks[start:end].tofile(out_dir / f"train_mask_{idx:04d}.bin")
        idx += 1
    packed[n_train:].tofile(out_dir / "val_0000.bin")
    masks[n_train:].tofile(out_dir / "val_mask_0000.bin")

    pad_frac = 1.0 - real_tokens / (n_seq * PACK_LEN)
    meta = {
        "seq_len": SEQ_LEN, "pack_len": PACK_LEN, "dtype": "uint16",
        "has_loss_mask": True, "multiturn": True,
        "tokenizer": TOKENIZER.name, "vocab_size": sp.get_piece_size(),
        "n_train_sequences": int(n_train), "n_val_sequences": int(n_seq - n_train),
        "n_train_shards": int(idx), "n_conversations": int(len(convs)),
        "assistant_token_frac": round(float(masks[:n_train].mean()), 4),
        "pad_frac": round(float(pad_frac), 4),
        "packing": "whole_conversation_binpack",
        "cross_conversation_attention": False,
        "dropped_too_long": int(dropped_long), "decontam_dropped": int(decontam),
        "source_counts": src_counts, "source_skips": skips,
        "synthetic_included": False,
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=True), encoding="utf-8")
    print(f"\nDone in {time.time()-t0:.0f}s: {n_train:,} train seqs, "
          f"assistant_frac={meta['assistant_token_frac']:.1%}, pad={pad_frac:.1%}")
    print(yaml.safe_dump(meta, sort_keys=True))


if __name__ == "__main__":
    main()
