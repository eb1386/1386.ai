#!/usr/bin/env python3
# plasma 1.2 pretrain corpus -- 30B tokens, seq 2048, tokenizer 1.2
#
# lessons from the 1.1 audit baked in:
#   - <|doc|> sentinel splitting for 1.2 downloads (blank-line splitting
#     shredded code files into ~300-char fragments and dropped 19% of code)
#   - mojibake scrub (corpus-borne a-hat artifacts got into 1.1's outputs)
#   - exact-hash dedup across all web sources (one boilerplate page appeared
#     224x in the 1.1 corpus; fineweb 10BT is a subset of 100BT)
#   - per-source token budgets with epoch caps; every source stays <=~1.7
#     epochs at the 30B target (mix table below)
#   - every mix path asserted non-empty before the build starts (the empty
#     code_clean.txt class of failure)
#
#   python scripts/build_corpus_1.2.py                # main 30B corpus
#   python scripts/build_corpus_1.2.py --mix anneal   # cooldown mix (~3B)

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.quality import filter_and_score  # noqa: E402
from src.data.safety import is_unsafe  # noqa: E402

RAW12 = ROOT / "data" / "raw_1.2"
RAW11 = ROOT / "data" / "raw_1.1"
DEDUP11 = ROOT / "data" / "deduped_1.1"
TOKENIZER = ROOT / "data" / "tokenizer_1.2.model"
SEQ_LEN = 2048
PACK_LEN = SEQ_LEN + 1
SENTINEL = "<|doc|>"

# name -> (paths, token_share, max_epochs, kind)
# token budgets audited against measured per-epoch yields; nothing exceeds
# ~1.7 epochs at 30B, so no source is memorized through repetition.
MAIN_MIX = {
    "fineweb_edu":     ([RAW12 / "fineweb_edu.txt", RAW11 / "fineweb_edu_hq.txt",
                         RAW11 / "fineweb_edu_hq2.txt"],                    0.26, 2, "web"),
    "fineweb_general": ([RAW11 / "fineweb_general.txt"],                    0.10, 2, "webgen"),
    "wikipedia":       ([RAW12 / "wiki_full.txt"],                          0.14, 2, "plain"),
    "code":            ([RAW12 / "code.txt", RAW11 / "code_clean.txt"], 0.17, 2, "code"),
    "books":           ([RAW12 / "books.txt"],                              0.10, 2, "plain"),
    "math":            ([RAW11 / "finemath_clean.txt", RAW12 / "finemath.txt"], 0.09, 2, "plain"),
    "stackexchange":   ([RAW11 / "stackexchange_clean.txt"],                0.05, 2, "stack"),
    "cosmopedia":      ([RAW11 / "cosmopedia_clean.txt", RAW12 / "cosmopedia.txt"], 0.07, 2, "plain"),
    "arxiv":           ([DEDUP11 / "arxiv_clean.txt"],                      0.02, 2, "plain"),
}

# cooldown: the highest-quality slice, upweighted for the wsd decay phase
ANNEAL_MIX = {
    "wikipedia":  ([RAW12 / "wiki_full.txt"],                          0.28, 1, "plain"),
    "cosmopedia": ([RAW11 / "cosmopedia_clean.txt", RAW12 / "cosmopedia.txt"], 0.22, 1, "plain"),
    "code_edu":   ([RAW12 / "code.txt"],                               0.16, 2, "code"),
    "math":       ([RAW11 / "finemath_clean.txt"],                     0.14, 1, "plain"),
    "books":      ([RAW12 / "books.txt"],                              0.12, 1, "plain"),
    "fineweb_edu": ([RAW12 / "fineweb_edu.txt"],                       0.08, 1, "web"),
}

_TAG = re.compile(r"<[^>]+>")
MOJIBAKE = {"â€™": "'", "â€œ": '"', "â€": '"', "â€“": "-", "â€”": "-",
            "â€¦": "...", "â€˜": "'"}


def scrub(text):
    for bad, good in MOJIBAKE.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def docs_from(path, min_chars=200):
    """sentinel-aware doc iterator; falls back to blank-line splitting for
    legacy 1.1 files that have no sentinels."""
    buf = []
    sentinel_seen = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.strip() == SENTINEL:
                sentinel_seen = True
                doc = "\n".join(buf).strip()
                if len(doc) >= min_chars:
                    yield scrub(doc)
                buf = []
            elif not line.strip() and not sentinel_seen:
                # legacy blank-line format
                if buf:
                    doc = "\n".join(buf).strip()
                    if len(doc) >= min_chars:
                        yield scrub(doc)
                    buf = []
            else:
                buf.append(line)
    if buf:
        doc = "\n".join(buf).strip()
        if len(doc) >= min_chars:
            yield scrub(doc)


def clean_stack(text):
    text = _TAG.sub("", text)
    import html as html_lib
    text = html_lib.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def one_pass(paths, kind, dedup_seen):
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        for doc in docs_from(path, 120 if kind in ("stack", "code") else 200):
            # exact-hash dedup across everything web-ish; code files can
            # legitimately repeat license headers, so hash more of them
            h = hashlib.md5(doc[:2000].encode("utf-8", "ignore")).digest()[:12]
            if h in dedup_seen:
                continue
            dedup_seen.add(h)
            if kind == "web":
                ok, _ = filter_and_score(doc, min_score=0.45)
                if ok:
                    yield doc
            elif kind == "webgen":
                if is_unsafe(doc):
                    continue
                ok, _ = filter_and_score(doc, min_score=0.30)
                if ok:
                    yield doc
            elif kind == "stack":
                doc = clean_stack(doc)
                if len(doc) >= 120:
                    yield doc
            else:
                yield doc


def epoch_iter(name, paths, kind, max_epochs, counter, dedup_seen):
    for ep in range(max_epochs):
        n = 0
        # dedup only on the first epoch; later epochs are deliberate repeats
        seen = dedup_seen if ep == 0 else set()
        for doc in one_pass(paths, kind, seen):
            n += 1
            yield doc
        counter[name] = ep + 1
        if n == 0:
            return
        print(f"  [{name}] epoch {ep+1}/{max_epochs} done ({n:,} docs)", flush=True)


def choose(active, weights, tok, total):
    if total <= 0:
        return active[0]
    best, best_def = active[0], -10.0
    for n in active:
        d = weights[n] - tok[n] / max(1, total)
        if d > best_def:
            best, best_def = n, d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", choices=["main", "anneal"], default="main")
    ap.add_argument("--target-tokens", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1386)
    args = ap.parse_args()

    mix = MAIN_MIX if args.mix == "main" else ANNEAL_MIX
    target = args.target_tokens or (30e9 if args.mix == "main" else 3e9)
    out_dir = Path(args.out_dir or (ROOT / "data" /
                   ("shards_1.2" if args.mix == "main" else "shards_1.2_anneal")))

    # every configured source must have at least one non-empty file: the
    # silent-empty-file class of failure cost 1.1 its code corpus
    missing = []
    for name, (paths, w, _, _) in mix.items():
        if w > 0 and not any(p.exists() and p.stat().st_size > 0 for p in paths):
            missing.append(f"  {name}: {[str(p) for p in paths]}")
    if missing:
        raise SystemExit("EMPTY/MISSING mix sources (run download_1.2_data.py):\n"
                         + "\n".join(missing))

    import sentencepiece as spm
    if not TOKENIZER.exists():
        raise SystemExit(f"tokenizer missing: {TOKENIZER} "
                         "(run scripts/train_tokenizer_1.2.py)")
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))
    eos = sp.eos_id()
    doc_id = sp.piece_to_id(SENTINEL)

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    names = [n for n in mix if mix[n][1] > 0]
    weights = {n: mix[n][1] for n in names}
    epochs_done = {}
    dedup_seen = set()
    iters = {n: epoch_iter(n, mix[n][0], mix[n][3], mix[n][2], epochs_done, dedup_seen)
             for n in names}
    tok_counts = {n: 0 for n in names}
    total = 0
    buf, train_seqs, val_seqs = [], [], []
    state = {"train_idx": 0, "train_sequences": 0, "val_sequences": 0}
    seq_i = 0
    t0 = time.time()

    def flush():
        nonlocal train_seqs, val_seqs
        if train_seqs:
            idx = rng.permutation(len(train_seqs))
            np.stack([train_seqs[i] for i in idx]).astype(np.uint16).tofile(
                out_dir / f"train_{state['train_idx']:04d}.bin")
            state["train_idx"] += 1
            state["train_sequences"] += len(train_seqs)
        if val_seqs:
            arr = np.stack(val_seqs).astype(np.uint16)
            with open(out_dir / "val_0000.bin", "ab") as f:
                arr.tofile(f)
            state["val_sequences"] += len(val_seqs)
        train_seqs, val_seqs = [], []

    active = list(names)
    while total < target and active:
        name = choose(active, weights, tok_counts, total)
        try:
            doc = next(iters[name])
        except StopIteration:
            active.remove(name)
            print(f"  [{name}] EXHAUSTED at {tok_counts[name]/1e9:.2f}B tokens",
                  flush=True)
            continue
        toks = sp.encode(doc) + [eos]
        tok_counts[name] += len(toks)
        total += len(toks)
        buf.extend(toks)
        while len(buf) >= PACK_LEN:
            seq = np.asarray(buf[:PACK_LEN], dtype=np.uint16)
            del buf[:PACK_LEN]
            seq_i += 1
            if seq_i % args.val_every == 0:
                val_seqs.append(seq)
            else:
                train_seqs.append(seq)
            if len(train_seqs) >= 40000:
                flush()
        if total and total % 500_000_000 < len(toks):
            shares = {n: round(tok_counts[n] / total, 3) for n in names}
            print(f"  {total/1e9:5.2f}B tokens | {time.time()-t0:6.0f}s | {shares}",
                  flush=True)

    flush()
    meta = {
        "tokenizer": TOKENIZER.name, "vocab_size": sp.get_piece_size(),
        "seq_len": SEQ_LEN, "pack_len": PACK_LEN, "dtype": "uint16",
        "total_tokens": int(total), "target_tokens": int(target),
        "mix": args.mix,
        "token_counts": {n: int(c) for n, c in tok_counts.items()},
        "token_shares": {n: round(c / max(total, 1), 4) for n, c in tok_counts.items()},
        "source_epochs": epochs_done,
        "n_train_sequences": state["train_sequences"],
        "n_val_sequences": state["val_sequences"],
        "dedup": "md5-2k exact, first epoch only",
        "sentinel_doc_splitting": True, "mojibake_scrubbed": True,
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=True),
                                       encoding="utf-8")
    print(f"\nDONE {total/1e9:.2f}B tokens, {state['train_sequences']:,} train seqs "
          f"in {(time.time()-t0)/3600:.1f}h -> {out_dir}")
    print(yaml.safe_dump(meta["token_shares"]))


if __name__ == "__main__":
    main()
