#!/usr/bin/env python3
# Build a targeted rescue SFT mix for Plasma 1.1.
#
# The failed SFT run overfit verbose reasoning templates and did not repair
# basic factual recall. This builder creates a smaller, cleaner mix weighted
# toward short direct answers, factual recall, arithmetic, and executable code.

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
RAW_11 = ROOT / "data" / "raw_1.1"
OUT_DIR = ROOT / "data" / "rescue_shards_1.1"
TOKENIZER = ROOT / "data" / "tokenizer_1.1.model"
SEQ_LEN = 1024

ROLE_RE = re.compile(r"(?:^|\n)(User|Assistant): ?")
MARKDOWN_ROLE_RE = re.compile(
    r"(?m)^\s*(?:[-*]\s*)?(?:#{1,6}\s*)?(?:\*\*)?"
    r"(User|Assistant)(?:\s*(?:turn|response|message))?"
    r"(?:\s*\d+)?\s*:(?:\*\*)?\s*"
)
TURN_HEADING_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?:\*\*)?Turn\s+\d+(?:\s*[-:][^\n]*)?(?:\*\*)?\s*$"
)
BAD_PATTERNS = (
    "i am a language model",
    "i am an ai",
    "i'm an ai",
    "as an ai",
    "as a language model",
    "i don't have access",
    "i do not have access",
    "i can't answer",
    "i cannot answer",
    "as a helpful assistant",
)

CAPITALS = {
    "Australia": "Canberra",
    "France": "Paris",
    "Germany": "Berlin",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Portugal": "Lisbon",
    "Canada": "Ottawa",
    "United States": "Washington, D.C.",
    "Mexico": "Mexico City",
    "Brazil": "Brasilia",
    "Argentina": "Buenos Aires",
    "Chile": "Santiago",
    "Japan": "Tokyo",
    "China": "Beijing",
    "India": "New Delhi",
    "South Korea": "Seoul",
    "Egypt": "Cairo",
    "South Africa": "Pretoria",
    "Nigeria": "Abuja",
    "Kenya": "Nairobi",
    "Russia": "Moscow",
    "Ukraine": "Kyiv",
    "Turkey": "Ankara",
    "Greece": "Athens",
    "Norway": "Oslo",
    "Sweden": "Stockholm",
    "Finland": "Helsinki",
    "Denmark": "Copenhagen",
    "Ireland": "Dublin",
    "Netherlands": "Amsterdam",
    "Belgium": "Brussels",
    "Switzerland": "Bern",
    "Austria": "Vienna",
    "Poland": "Warsaw",
}

ELEMENTS = {
    "hydrogen": "H",
    "helium": "He",
    "carbon": "C",
    "nitrogen": "N",
    "oxygen": "O",
    "sodium": "Na",
    "magnesium": "Mg",
    "aluminum": "Al",
    "silicon": "Si",
    "phosphorus": "P",
    "sulfur": "S",
    "chlorine": "Cl",
    "potassium": "K",
    "calcium": "Ca",
    "iron": "Fe",
    "copper": "Cu",
    "zinc": "Zn",
    "silver": "Ag",
    "tin": "Sn",
    "iodine": "I",
    "platinum": "Pt",
    "gold": "Au",
    "mercury": "Hg",
    "lead": "Pb",
    "uranium": "U",
}

FACTS = [
    ("Who was the first president of the United States?", "George Washington."),
    ("Who was the first person to walk on the moon?", "Neil Armstrong."),
    ("What year did World War II end?", "1945."),
    ("What is the largest planet in our solar system?", "Jupiter."),
    ("What planet is known as the Red Planet?", "Mars."),
    ("What gas do plants take in during photosynthesis?", "Carbon dioxide."),
    ("What gas do plants release during photosynthesis?", "Oxygen."),
    ("What is the boiling point of water at sea level in Celsius?", "100 degrees Celsius."),
    ("How many continents are there?", "Seven."),
    ("How many planets are in the solar system?", "Eight."),
    ("What is the main language used for web page structure?", "HTML."),
    ("What keyword defines a function in Python?", "def."),
    ("What is the file extension for Python files?", ".py."),
    ("What is the speed of light approximately?", "About 300,000 kilometers per second."),
    ("What is H2O commonly called?", "Water."),
]

CODE_EXAMPLES = [
    (
        "Write a Python function that checks if a number is prime.",
        "```python\n"
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n"
        "```",
    ),
    (
        "Write a Python function that returns the sum of two numbers.",
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```",
    ),
    (
        "Write a Python function that reverses a list.",
        "```python\n"
        "def reverse_list(items):\n"
        "    return items[::-1]\n"
        "```",
    ),
    (
        "Write a Python function that checks whether a string is a palindrome.",
        "```python\n"
        "def is_palindrome(text):\n"
        "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
        "    return cleaned == cleaned[::-1]\n"
        "```",
    ),
]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = TURN_HEADING_RE.sub("", text)
    text = MARKDOWN_ROLE_RE.sub(lambda m: f"{m.group(1)}: ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def bad_sample(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in BAD_PATTERNS):
        return True
    if "Assistant:" not in text or "User:" not in text:
        return True
    if len(text) > 7000:
        return True
    return False


def fmt(user: str, assistant: str, source: str) -> dict:
    return {"text": f"User: {user.strip()}\nAssistant: {assistant.strip()}", "source": source}


def targeted_records(repeats: int) -> list[dict]:
    rows = []
    templates = [
        ("What is the capital of {k}?", "{v}."),
        ("Name the capital city of {k}.", "{v}."),
        ("{k}'s capital is what?", "{v}."),
    ]
    for _ in range(repeats):
        for country, capital in CAPITALS.items():
            for q, a in templates:
                rows.append(fmt(q.format(k=country, v=capital), a.format(k=country, v=capital), "target_capitals"))

        for name, symbol in ELEMENTS.items():
            rows.append(fmt(f"What is the chemical symbol for {name}?", f"{symbol}.", "target_elements"))
            rows.append(fmt(f"The chemical symbol for {name} is what?", f"{symbol}.", "target_elements"))
            rows.append(fmt(f"Answer with only the symbol: {name}", symbol, "target_elements_short"))

        for question, answer in FACTS:
            rows.append(fmt(question, answer, "target_facts"))
            rows.append(fmt(f"Answer directly: {question}", answer, "target_facts_direct"))

        for question, answer in CODE_EXAMPLES:
            rows.append(fmt(question, answer, "target_code"))
            rows.append(fmt(f"Answer with clean Python only: {question}", answer, "target_code_direct"))

    return rows


def arithmetic_records(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.randint(0, 200)
        b = rng.randint(0, 200)
        rows.append(fmt(f"What is {a} + {b}?", str(a + b), "target_math"))
        if a >= b:
            rows.append(fmt(f"What is {a} - {b}?", str(a - b), "target_math"))
        x = rng.randint(2, 20)
        y = rng.randint(2, 20)
        rows.append(fmt(f"What is {x} x {y}?", str(x * y), "target_math"))

    word_problems = [
        (60, 2.5),
        (50, 3),
        (70, 2),
        (45, 4),
        (30, 1.5),
        (80, 1.25),
        (55, 2),
        (65, 3),
    ]
    for _ in range(400):
        for speed, hours in word_problems:
            dist = int(speed * hours) if float(speed * hours).is_integer() else speed * hours
            rows.append(fmt(
                f"If a train travels at {speed} mph for {hours} hours, how far does it go?",
                f"{dist} miles.",
                "target_word_math",
            ))
    return rows


def short_behavior_records() -> list[dict]:
    return [
        fmt("What color is the sky?", "Blue.", "target_short"),
        fmt("List exactly five common fruits.", "Apple, banana, orange, grape, and pear.", "target_short"),
        fmt("Explain photosynthesis in one sentence.", "Photosynthesis is how plants use light, water, and carbon dioxide to make sugar and release oxygen.", "target_short"),
        fmt("What is the difference between weather and climate?", "Weather is short-term atmospheric conditions; climate is the long-term pattern of weather in a region.", "target_short"),
        fmt("Write a short poem about the ocean.", "Blue waves rise and fall,\nMoonlight rests on moving tides,\nThe sea breathes softly.", "target_short"),
    ] * 2000


def code_drill_records(repeats: int = 600) -> list[dict]:
    rows = []
    for _ in range(repeats):
        for question, answer in CODE_EXAMPLES:
            rows.append(fmt(question, answer, "target_code_drill"))
            rows.append(fmt(f"Return only valid Python code. {question}", answer, "target_code_drill"))
    return rows


def iter_public_records(seed: int, limits: dict[str, int]):
    rng = random.Random(seed)
    files = {
        "instruct_no_robots": RAW_11 / "instruct_no_robots.jsonl",
        "instruct_capybara": RAW_11 / "instruct_capybara.jsonl",
        "instruct_metamath": RAW_11 / "instruct_metamath.jsonl",
        "instruct_wizardlm": RAW_11 / "instruct_wizardlm.jsonl",
        "instruct_ultrachat": RAW_11 / "instruct_ultrachat.jsonl",
        "instruct_slimorca": RAW_11 / "instruct_slimorca.jsonl",
        "synthetic_instruct": RAW_11 / "synthetic_instruct.jsonl",
    }
    for source, path in files.items():
        limit = limits.get(source, 0)
        if limit <= 0 or not path.exists():
            continue
        kept = 0
        seen = 0
        # Deterministic reservoir-ish scan: accept random rows until limit.
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                seen += 1
                try:
                    text = normalize_text(json.loads(line)["text"])
                except Exception:
                    continue
                if bad_sample(text):
                    continue
                # Prefer concise samples; keep some longer multi-turn rows.
                words = len(text.split())
                if words > 900:
                    continue
                accept_p = 0.55 if source in {"instruct_metamath", "instruct_no_robots", "synthetic_instruct"} else 0.35
                if rng.random() > accept_p:
                    continue
                yield {"text": text, "source": source}
                kept += 1
                if kept >= limit:
                    break
        print(f"  {source}: kept {kept:,} / scanned {seen:,}")


def build_mask(text: str, sp: spm.SentencePieceProcessor):
    boundaries = []
    for m in ROLE_RE.finditer(text):
        role_start = m.start(1)
        label_end = m.end()
        role = "user" if m.group(1) == "User" else "assistant"
        boundaries.append((role, role_start, label_end))
    if not boundaries:
        return None, None

    tokens = []
    mask_bits = []
    if boundaries[0][1] > 0:
        prefix = text[: boundaries[0][1]]
        ptoks = sp.encode(prefix, out_type=int)
        tokens.extend(ptoks)
        mask_bits.extend([0] * len(ptoks))

    for i, (role, role_start, label_end) in enumerate(boundaries):
        next_start = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
        label = text[role_start:label_end]
        ltoks = sp.encode(label, out_type=int)
        tokens.extend(ltoks)
        mask_bits.extend([0] * len(ltoks))

        content = text[label_end:next_start]
        ctoks = sp.encode(content, out_type=int)
        bit = 1 if role == "assistant" else 0
        tokens.extend(ctoks)
        mask_bits.extend([bit] * len(ctoks))

    if len(tokens) < 4 or not any(mask_bits):
        return None, None
    return tokens, np.asarray(mask_bits, dtype=np.uint8)


def write_shards(records: list[dict], out_dir: Path, seed: int):
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))
    eos_id = sp.eos_id()
    pack_len = SEQ_LEN + 1

    rng = random.Random(seed)
    rng.shuffle(records)

    conv_tokens = []
    conv_masks = []
    source_counts = {}
    skipped = 0
    total_tokens = 0
    for rec in records:
        text = normalize_text(rec["text"])
        if bad_sample(text):
            skipped += 1
            continue
        toks, mask = build_mask(text, sp)
        if toks is None:
            skipped += 1
            continue
        toks.append(eos_id)
        mask = np.append(mask, 1)
        conv_tokens.append(np.asarray(toks, dtype=np.uint16))
        conv_masks.append(mask.astype(np.uint8))
        total_tokens += len(toks)
        source = rec.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    if not conv_tokens:
        raise RuntimeError("no rescue conversations survived filtering")

    all_tokens = np.concatenate(conv_tokens)
    all_masks = np.concatenate(conv_masks)
    n_seq = len(all_tokens) // pack_len
    trimmed = n_seq * pack_len
    packed_tokens = all_tokens[:trimmed].reshape(n_seq, pack_len)
    packed_masks = all_masks[:trimmed].reshape(n_seq, pack_len)

    rng_np = np.random.default_rng(seed)
    perm = rng_np.permutation(n_seq)
    packed_tokens = packed_tokens[perm]
    packed_masks = packed_masks[perm]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_val = max(1, n_seq // 20)
    n_train = n_seq - n_val
    seqs_per_shard = 100_000
    shard_idx = 0
    for start in range(0, n_train, seqs_per_shard):
        end = min(start + seqs_per_shard, n_train)
        (out_dir / f"train_{shard_idx:04d}.bin").write_bytes(packed_tokens[start:end].tobytes())
        (out_dir / f"train_mask_{shard_idx:04d}.bin").write_bytes(packed_masks[start:end].tobytes())
        shard_idx += 1
    (out_dir / "val_0000.bin").write_bytes(packed_tokens[n_train:].tobytes())
    (out_dir / "val_mask_0000.bin").write_bytes(packed_masks[n_train:].tobytes())

    meta = {
        "seq_len": SEQ_LEN,
        "pack_len": pack_len,
        "dtype": "uint16",
        "has_loss_mask": True,
        "n_conversations": len(conv_tokens),
        "skipped": skipped,
        "total_tokens": int(total_tokens),
        "n_sequences": int(n_seq),
        "n_train_sequences": int(n_train),
        "n_val_sequences": int(n_val),
        "n_train_shards": int(shard_idx),
        "source_counts": source_counts,
        "assistant_mask_pct": float(packed_masks[:n_train].sum() / max(1, n_train * pack_len) * 100),
    }
    with open(out_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=True)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-repeats", type=int, default=80)
    parser.add_argument("--math-samples", type=int, default=8000)
    args = parser.parse_args()

    t0 = time.time()
    records = []
    records.extend(targeted_records(args.target_repeats))
    records.extend(arithmetic_records(args.math_samples, args.seed))
    records.extend(short_behavior_records())
    records.extend(code_drill_records())

    limits = {
        "instruct_no_robots": 8_000,
        "instruct_capybara": 8_000,
        "instruct_metamath": 35_000,
        "instruct_wizardlm": 12_000,
        "instruct_ultrachat": 20_000,
        "instruct_slimorca": 20_000,
        "synthetic_instruct": 12_000,
    }
    records.extend(iter_public_records(args.seed, limits))

    print(f"Total candidate records: {len(records):,}")
    meta = write_shards(records, Path(args.out_dir), args.seed)
    print("\nRescue SFT shards built:")
    for k, v in meta.items():
        if k != "source_counts":
            print(f"  {k}: {v}")
    print("  source_counts:")
    for k, v in sorted(meta["source_counts"].items()):
        print(f"    {k}: {v:,}")
    print(f"  elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
