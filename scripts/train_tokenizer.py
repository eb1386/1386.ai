#!/usr/bin/env python3
# train tokenizer

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sample_corpus(raw_dir: Path, output_path: Path, target_mb: int = 2000):
    """Sample from all sources for tokenizer training."""
    source_configs = [
        ("fineweb_edu_hq.txt", 0.50),
        ("wikipedia_clean.txt", 0.15),
        ("stackexchange_clean.txt", 0.10),
        ("code_clean.txt", 0.10),
        ("books_clean.txt", 0.10),
        ("arxiv_clean.txt", 0.05),
    ]

    v4_raw = ROOT / "data" / "raw_v4"
    fallback_sources = [
        (v4_raw / "pretrain_corpus.txt", 0.30),
        (v4_raw / "fineweb_edu_corpus.txt", 0.20),
    ]

    sources = []
    for filename, weight in source_configs:
        path = raw_dir / filename
        if path.exists() and path.stat().st_size > 0:
            sources.append((path, weight))

    if not sources:
        print("  No 1.1 sources, using fallback...")
        for path, weight in fallback_sources:
            if path.exists() and path.stat().st_size > 0:
                sources.append((path, weight))

    if not sources:
        print("ERROR: No corpus files found!")
        sys.exit(1)

    total_weight = sum(w for _, w in sources)
    sources = [(p, w / total_weight) for p, w in sources]

    target_bytes = target_mb * 1_000_000
    print(f"Sampling {target_mb} MB from {len(sources)} sources...")

    total_written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for path, weight in sources:
            source_target = int(target_bytes * weight)
            source_written = 0
            print(f"  {path.name}: {source_target / 1e6:.0f} MB ({weight:.0%})...")

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    source_written += len(line.encode("utf-8"))
                    total_written += len(line.encode("utf-8"))
                    if source_written >= source_target:
                        break

            print(f"    Sampled {source_written / 1e6:.1f} MB")

    print(f"\n  Total: {total_written / 1e6:.1f} MB")
    return output_path


def train_tokenizer(corpus_path: Path, output_prefix: str,
                    vocab_size: int = 48000, character_coverage: float = 0.9999):
    """Train SentencePiece BPE tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: pip install sentencepiece")
        sys.exit(1)

    print(f"\nTraining SentencePiece BPE...")
    print(f"  Vocab: {vocab_size:,}")
    print(f"  Input: {corpus_path} ({corpus_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  Output: {output_prefix}.model")

    t0 = time.time()

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
        pad_id=3, unk_id=0, bos_id=1, eos_id=2,
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=True,
        max_sentencepiece_length=16,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
    )

    print(f"  Done in {(time.time() - t0) / 60:.1f} min")
    return f"{output_prefix}.model"


def compare_tokenizers(old_path: str, new_path: str, test_texts: list[str] | None = None):
    """Compare old vs new tokenizer."""
    import sentencepiece as spm

    old_sp = spm.SentencePieceProcessor()
    old_sp.load(old_path)
    new_sp = spm.SentencePieceProcessor()
    new_sp.load(new_path)

    if test_texts is None:
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In 1776, the United States Declaration of Independence was adopted.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "The integral of x squared from 0 to 1 is equal to 1/3.",
            "Question: What is the capital of France?\nAnswer: The capital of France is Paris.",
            "According to the paper by Vaswani et al. (2017), the Transformer architecture uses self-attention mechanisms.",
        ]

    print(f"\nComparison:")
    print(f"  Old vocab: {old_sp.get_piece_size():,}")
    print(f"  New vocab: {new_sp.get_piece_size():,}\n")

    total_old = 0
    total_new = 0
    for text in test_texts:
        old_tokens = old_sp.encode(text, out_type=int)
        new_tokens = new_sp.encode(text, out_type=int)
        total_old += len(old_tokens)
        total_new += len(new_tokens)
        savings = (1 - len(new_tokens) / len(old_tokens)) * 100
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"  {preview}")
        print(f"    old: {len(old_tokens)}, new: {len(new_tokens)} ({savings:+.1f}%)")

    overall = (1 - total_new / total_old) * 100
    print(f"\n  Overall: {total_old} -> {total_new} ({overall:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer for Plasma 1.1")
    parser.add_argument("--raw-dir", type=str, default=str(ROOT / "data" / "deduped_1.1"))
    parser.add_argument("--output", type=str, default=str(ROOT / "data" / "tokenizer_1.1"))
    parser.add_argument("--vocab-size", type=int, default=48000)
    parser.add_argument("--sample-mb", type=int, default=2000)
    parser.add_argument("--compare-old", type=str, default=str(ROOT / "data" / "tokenizer_v4.model"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     dir=str(ROOT / "data")) as tmp:
        corpus_path = Path(tmp.name)

    try:
        sample_corpus(raw_dir, corpus_path, target_mb=args.sample_mb)
        model_path = train_tokenizer(corpus_path, args.output, vocab_size=args.vocab_size)

        old_tok = args.compare_old
        if Path(old_tok).exists():
            compare_tokenizers(old_tok, model_path)

        print(f"\nSaved: {model_path}")
    finally:
        if corpus_path.exists():
            corpus_path.unlink()


if __name__ == "__main__":
    main()
