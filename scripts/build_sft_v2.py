#!/usr/bin/env python3
# clean broad SFT for plasma 1.1 from real public instruction data.
# diverse, decontaminated vs benchmark, assistant-only loss masking, EOS taught.
# drops the hallucination-prone synthetic set and any benchmark-overlapping rows.

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
OUT = ROOT / "data" / "sft_shards_1.1_v2"
TOKENIZER = ROOT / "data" / "tokenizer_1.1.model"
SEQ_LEN = 1024
PACK_LEN = SEQ_LEN + 1

# source -> max records kept (caps keep the mix balanced; big sources dominate otherwise)
SOURCES = {
    "instruct_slimorca":  120000,
    "instruct_ultrachat":  80000,
    "instruct_wizardlm":   50000,
    "instruct_metamath":   45000,
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
    "i cannot answer", "i can't answer",
    "i cannot provide", "i'm just an ai",
)


def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def bad_sample(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in BAD_PATTERNS):
        return True
    if len(text) < 24:
        return True
    return False


def first_user(text: str) -> str:
    m = re.search(r"(?:^|\n)User: ?(.*?)(?:\nAssistant:|$)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def build_mask(text: str, sp):
    """tokens + per-token mask; 1 only on assistant content, 0 on roles/user."""
    boundaries = []
    for m in ROLE_RE.finditer(text):
        role = "user" if m.group(1) == "User" else "assistant"
        boundaries.append((role, m.start(1), m.end()))
    if not boundaries:
        return None, None

    tokens, bits = [], []
    if boundaries[0][1] > 0:
        ptoks = sp.encode(text[: boundaries[0][1]], out_type=int)
        tokens.extend(ptoks)
        bits.extend([0] * len(ptoks))

    for i, (role, role_start, label_end) in enumerate(boundaries):
        next_start = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
        ltoks = sp.encode(text[role_start:label_end], out_type=int)
        tokens.extend(ltoks)
        bits.extend([0] * len(ltoks))
        ctoks = sp.encode(text[label_end:next_start], out_type=int)
        bit = 1 if role == "assistant" else 0
        tokens.extend(ctoks)
        bits.extend([bit] * len(ctoks))

    if len(tokens) < 8 or not any(bits):
        return None, None
    return tokens, np.asarray(bits, dtype=np.uint8)


def load_source(name: str, cap: int, rng: random.Random):
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        print(f"  [warn] missing {path}")
        return []
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("text", "")
            if t:
                rows.append(t)
    rng.shuffle(rows)
    return rows[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--seed", type=int, default=20260607)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and (out_dir / "meta.yaml").exists() and not args.force:
        print(f"[skip] existing sft shards: {out_dir}")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))
    eos_id = sp.eos_id()
    rng = random.Random(args.seed)
    exclude = all_prompt_strings()

    t0 = time.time()
    records = []  # (text, source)
    for name, cap in SOURCES.items():
        rows = load_source(name, cap, rng)
        for t in rows:
            records.append((t, name))
        print(f"  loaded {name}: {len(rows):,}", flush=True)

    rng.shuffle(records)

    conv_tokens, conv_masks = [], []
    source_counts, source_skips = {}, {}
    decontam_dropped = 0
    total_tokens = 0
    for text, source in records:
        text = normalize_text(text)
        if bad_sample(text):
            source_skips[source] = source_skips.get(source, 0) + 1
            continue
        # decontamination vs benchmark
        fu = norm_prompt(first_user(text))
        if fu in exclude:
            decontam_dropped += 1
            continue
        toks, mask = build_mask(text, sp)
        if toks is None:
            source_skips[source] = source_skips.get(source, 0) + 1
            continue
        toks.append(eos_id)
        mask = np.append(mask, 1).astype(np.uint8)
        conv_tokens.append(np.asarray(toks, dtype=np.uint16))
        conv_masks.append(mask)
        total_tokens += len(toks)
        source_counts[source] = source_counts.get(source, 0) + 1

    if not conv_tokens:
        raise SystemExit("no sft conversations survived filtering")

    all_tokens = np.concatenate(conv_tokens)
    all_masks = np.concatenate(conv_masks)
    n_seq = len(all_tokens) // PACK_LEN
    trim = n_seq * PACK_LEN
    packed = all_tokens[:trim].reshape(n_seq, PACK_LEN)
    packed_m = all_masks[:trim].reshape(n_seq, PACK_LEN)

    rng_np = np.random.default_rng(args.seed)
    perm = rng_np.permutation(n_seq)
    packed, packed_m = packed[perm], packed_m[perm]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(n_seq * args.val_frac))
    n_train = n_seq - n_val
    seqs_per_shard = 100_000

    train_idx = 0
    for start in range(0, n_train, seqs_per_shard):
        end = min(start + seqs_per_shard, n_train)
        packed[start:end].astype(np.uint16).tofile(out_dir / f"train_{train_idx:04d}.bin")
        packed_m[start:end].astype(np.uint8).tofile(out_dir / f"train_mask_{train_idx:04d}.bin")
        train_idx += 1
    packed[n_train:].astype(np.uint16).tofile(out_dir / "val_0000.bin")
    packed_m[n_train:].astype(np.uint8).tofile(out_dir / "val_mask_0000.bin")

    assistant_frac = float(packed_m[:n_train].mean())
    meta = {
        "seq_len": SEQ_LEN, "pack_len": PACK_LEN, "dtype": "uint16",
        "has_loss_mask": True, "multiturn": True,
        "tokenizer": TOKENIZER.name, "vocab_size": sp.get_piece_size(),
        "n_train_sequences": int(n_train), "n_val_sequences": int(n_seq - n_train),
        "n_train_shards": int(train_idx), "total_tokens": int(total_tokens),
        "assistant_token_frac": round(assistant_frac, 4),
        "source_counts": source_counts, "source_skips": source_skips,
        "decontam_dropped": int(decontam_dropped),
        "synthetic_included": False,
    }
    with open(out_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=True)
    print(f"\nDone in {time.time()-t0:.0f}s. train_seq={n_train:,} val_seq={n_seq-n_train:,} "
          f"tokens={total_tokens/1e6:.1f}M assistant_frac={assistant_frac:.1%} "
          f"decontam_dropped={decontam_dropped}", flush=True)
    print(yaml.safe_dump(meta, sort_keys=True))


if __name__ == "__main__":
    main()
