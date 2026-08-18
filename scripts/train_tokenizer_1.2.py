#!/usr/bin/env python3
# plasma 1.2 tokenizer
#
# the 1.1 tokenizer had three structural defects (audit, aug 2026):
#   1. remove_extra_whitespaces=True made encoding LOSSY: '    return x'
#      encoded identically to 'return x' -- indentation unrepresentable, so
#      the model could never emit valid python.
#   2. no newline piece: every \n was a byte-fallback token that also broke
#      the next word's merge ('User' -> 'U'+'ser'), causing the train/serve
#      template seam.
#   3. its training sample contained ZERO code (code_clean.txt was 0 bytes
#      and the sampler silently renormalized weights).
# fixes here: lossless whitespace, \n and \t as first-class symbols, atomic
# chat-role tokens, code required in the sample (loud failure otherwise).
#
#   python scripts/train_tokenizer_1.2.py            # sample + train + verify
#   python scripts/train_tokenizer_1.2.py --verify   # verify existing model

import argparse
import os
import sys
import time
from pathlib import Path

# windows console defaults to cp1252; piece strings contain U+2581
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw_1.2"
RAW_11 = ROOT / "data" / "raw_1.1"
OUT_PREFIX = str(ROOT / "data" / "tokenizer_1.2")
SAMPLE = ROOT / "data" / "tokenizer_1.2_sample.txt"

# atomic chat markers: single un-splittable ids, so the sft template can
# never fragment and stop detection is one integer compare
SPECIALS = ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>", "<|doc|>"]

# (path, weight, required) -- required sources hard-fail if missing/empty
SOURCES = [
    (RAW / "fineweb_edu.txt",    0.40, False),
    (RAW_11 / "fineweb_edu_hq.txt", 0.40, False),   # fallback pre-download
    (RAW / "wiki_full.txt", 0.15, False),
    (RAW_11 / "wikipedia_clean.txt", 0.15, False),
    (RAW / "code.txt", 0.20, True),
    (RAW / "books.txt", 0.10, False),
    (RAW_11 / "finemath_clean.txt", 0.08, False),
    (RAW_11 / "arxiv_clean.txt", 0.07, False),
]


def resolve_sources():
    """prefer 1.2 downloads, fall back to 1.1 files for the same slot"""
    picked, seen_slots = [], set()
    for path, weight, required in SOURCES:
        slot = round(weight, 3)
        key = (slot, path.name.split("_")[0])
        if key in seen_slots:
            continue
        if path.exists() and path.stat().st_size > 1_000_000:
            picked.append((path, weight, required))
            seen_slots.add(key)
        elif required:
            raise SystemExit(
                f"REQUIRED tokenizer source missing or empty: {path}\n"
                f"run scripts/download_1.2_data.py first -- training a "
                f"tokenizer with no code is how 1.1 went wrong.")
    if not picked:
        raise SystemExit("no tokenizer sources found")
    total = sum(w for _, w, _ in picked)
    picked = [(p, w / total, r) for p, w, r in picked]
    print("effective sample weights (renormalized, all sources verified non-empty):")
    for p, w, _ in picked:
        print(f"  {w:5.1%}  {p}")
    return picked


def sample_corpus(target_mb=2400):
    sources = resolve_sources()
    target = target_mb * 1_000_000
    written = 0
    with open(SAMPLE, "w", encoding="utf-8") as out:
        for path, weight, _ in sources:
            goal = int(target * weight)
            got = 0
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip() == "<|doc|>":
                        continue        # sentinel lines are structure, not text
                    out.write(line)
                    got += len(line.encode("utf-8", "ignore"))
                    if got >= goal:
                        break
            written += got
            print(f"  sampled {got/1e6:6.0f} MB from {path.name}")
    print(f"  total {written/1e6:.0f} MB -> {SAMPLE}")


def train():
    import sentencepiece as spm
    print("training sentencepiece bpe (lossless whitespace, 48k)...")
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=str(SAMPLE),
        model_prefix=OUT_PREFIX,
        vocab_size=48000,
        model_type="bpe",
        character_coverage=0.9999,
        byte_fallback=True,
        split_digits=True,
        # the 1.1 killers, fixed:
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        # no phantom prefix: span-composed ids must equal full-string ids,
        # or the train/serve seam bug returns through the back door
        add_dummy_prefix=False,
        normalization_rule_name="identity",
        user_defined_symbols=SPECIALS + ["\n", "\t"],
        pad_id=3, unk_id=0, bos_id=1, eos_id=2,
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=True,
        max_sentencepiece_length=16,
        input_sentence_size=12_000_000,
        shuffle_input_sentence=True,
    )
    print(f"  done in {(time.time()-t0)/60:.1f} min -> {OUT_PREFIX}.model")


def verify():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(OUT_PREFIX + ".model")
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        ok = ok and cond

    code = "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)"
    check("indentation lossless", sp.decode(sp.encode(code)) == code)
    check("whitespace run kept", sp.decode(sp.encode("a    b")) == "a    b")
    for s in SPECIALS:
        pid = sp.piece_to_id(s)
        check(f"atomic {s}", pid > 0 and sp.id_to_piece(pid) == s, f"id={pid}")
    nl = sp.encode("\n")
    check("newline single token", len(nl) == 1, f"ids={nl}")
    d = sp.encode("1234")
    check("digits split", len(d) >= 4, f"n={len(d)}")
    tpl = sp.encode("<|user|>What is 2+2?<|end|>\n<|assistant|>")
    pieces = [sp.id_to_piece(t) for t in tpl]
    check("template stable", pieces[0] == "<|user|>" and "<|assistant|>" in pieces,
          str(pieces[:6]))
    # efficiency numbers for the record
    py = "for i in range(10):\n    print(i * 2)\n"
    prose = "The quick brown fox jumps over the lazy dog near the river bank."
    print(f"  code tok/char {len(sp.encode(py))/len(py):.3f} | "
          f"prose tok/char {len(sp.encode(prose))/len(prose):.3f}")
    if not ok:
        raise SystemExit("tokenizer verification FAILED")
    print("  tokenizer 1.2 verified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sample-mb", type=int, default=2400)
    args = ap.parse_args()
    if args.verify:
        verify()
        return
    if Path(OUT_PREFIX + ".model").exists():
        print(f"[skip] {OUT_PREFIX}.model exists")
        verify()
        return
    sample_corpus(args.sample_mb)
    train()
    verify()
    SAMPLE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
