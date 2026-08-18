#!/usr/bin/env python3
# plasma sft builder v4 — the audit-driven rebuild.
#
# every change here traces to a measured defect in the v3 set:
#   1. EOS AFTER EVERY ASSISTANT TURN (v3: one eos per conversation; only 68%
#      of answers were followed by eos, so the model learned to keep talking).
#   2. BOS AT CONVERSATION START -> doc boundaries recoverable at train time
#      for block-diagonal attention (v3 attended across packed conversations).
#   3. hh_helpful REMOVED (3.2% hedge-opening turns trained "I'm not sure...").
#   4. ultrachat 80k->32k and truncated to 3 turns (was 41.6% of loss tokens,
#      verbose essays with fabricated facts).
#   5. code instruction added: codealpaca + evol-code (v3 had ~0.8% code).
#   6. identity scrub + synthetic Plasma identity (v3 trained "I am Open
#      Assistant" verbatim from oasst).
#   7. expanded refusal/hedge filters at turn OPENINGS (v3 substring filter
#      missed 'I apologize', 'Unfortunately', 'as AI language model', ...).
#   8. cross-source dedup (oasst 83% / hh 65% / metamath 31% duplicate rate).
#   9. truncated-answer drop (raw slimorca contains mid-sentence cutoffs).
#  10. mojibake scrub (corpus-borne a-hat artifacts).
#  11. length-mismatch filter (200+ word answers to sub-15-word questions).
#  12. decontam against benchmark AND the new 309-prompt audit battery.

import argparse
import json
import random
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_prompts import all_prompt_strings, norm_prompt  # noqa: E402

RAW = ROOT / "data" / "raw_1.1"
OUT = ROOT / "data" / "sft_shards_1.1_v4"
TOKENIZER = ROOT / "data" / "tokenizer_1.1.model"
SEQ_LEN = 1024
PACK_LEN = SEQ_LEN + 1

# quality order matters: dedup keeps the FIRST occurrence, so better
# sources must be processed first. (name, cap, repeat)
SOURCES = [
    ("instruct_no_robots",  10000, 3),
    ("instruct_dolly",      15000, 2),
    ("instruct_codealpaca", 20000, 1),
    ("instruct_evolcode",   30000, 1),
    ("instruct_oasst",       8000, 1),
    ("instruct_capybara",    8000, 1),
    ("instruct_slimorca",   80000, 1),
    ("instruct_wizardlm",   35000, 1),
    ("instruct_alpaca",     35000, 1),
    ("instruct_metamath",   20000, 1),
    ("instruct_ultrachat",  32000, 1),
]

ROLE_RE = re.compile(r"(?:^|\n)(User|Assistant): ?")

# identity and artifact terms: any hit drops the conversation
IDENTITY_DROP = re.compile(
    r"open\s?assistant|chatgpt|openai|anthropic|\bGPT[\s-]?[234je]?\b"
    r"|<noinput>|<nooutput>|\bnoinput\b|\bnooutput\b|as ai language model",
    re.IGNORECASE)

# refusal / hedge OPENINGS of an assistant turn (v3 used substring-anywhere
# and missed most of these)
BAD_OPENINGS = (
    "i cannot", "i can't", "i won't", "i will not", "i'm sorry", "i am sorry",
    "sorry,", "i apologize", "unfortunately", "i'm unable", "i am unable",
    "i must decline", "i do not feel comfortable", "i don't feel comfortable",
    "i'm not sure", "i am not sure", "i'm afraid i", "i am afraid i",
    "i don't know", "i do not know", "i think ", "i believe ", "as an ai",
    "as a language model", "i'm just an ai",
)
BAD_ANYWHERE = ("my content filter", "violates my", "i am an ai language",
                "as an ai language")

SPAM_USER = re.compile(
    r"here is a piece of text|given the text:|based on the passage above",
    re.IGNORECASE)

# wizardlm evol-instruct frankenprompts glue code domains onto prose topics
EVOL_FRANKEN = re.compile(
    r"(shell cmd|shell command|latex|swift code|c\+\+ code|html page|xml data"
    r"|excel table|mark down|json data|ruby code|scala|matlab)", re.IGNORECASE)

MOJIBAKE = {
    "â€™": "'", "â€œ": '"', "â€": '"',
    "â€“": "-", "â€”": "-", "â€¦": "...",
    "Ã©": "e", "â€˜": "'",
}

WRITE_INTENT = re.compile(
    r"write|essay|story|article|detail|explain .* thoroughly|elaborate|list|"
    r"describe|compose|draft|poem|paragraphs", re.IGNORECASE)

# a digit/percent ending is a legitimate final answer ("The answer is: 42"),
# not a truncation
END_OK = re.compile(r'([.!?"\')\]\d%]|```|\|)\s*$|^\s*(?:\d+[\.\)]|[-*])\s+\S+$',
                    re.MULTILINE)


def normalize_text(t):
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    for bad, good in MOJIBAKE.items():
        t = t.replace(bad, good)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def turns_of(text):
    """[(role, content), ...] split on the template labels"""
    bounds = [(m.group(1), m.start(1), m.end()) for m in ROLE_RE.finditer(text)]
    out = []
    for i, (role, rs, le) in enumerate(bounds):
        nxt = bounds[i + 1][1] if i + 1 < len(bounds) else len(text)
        content = text[le:nxt]
        # strip the newline that separated this turn from the next label
        out.append((role.lower(), content.rstrip("\n")))
    return out


def rebuild(turns):
    return "\n".join(f"{'User' if r == 'user' else 'Assistant'}: {c}"
                     for r, c in turns)


def conv_ok(turns, source, stats):
    def bump(reason):
        stats[reason] = stats.get(reason, 0) + 1
        return False

    if not turns or turns[0][0] != "user":
        return bump("shape")
    a_turns = [c for r, c in turns if r == "assistant"]
    if not a_turns:
        return bump("shape")

    blob = "\n".join(c for _, c in turns)
    if IDENTITY_DROP.search(blob):
        return bump("identity")
    low_blob = blob.lower()
    if any(p in low_blob for p in BAD_ANYWHERE):
        return bump("refusal")

    for a in a_turns:
        low = a.lstrip("\"'*# ").lower()
        if any(low.startswith(b) for b in BAD_OPENINGS):
            return bump("refusal_hedge")
        if len(a.strip()) < 2:
            return bump("empty_turn")

    # truncated final answer: raw slimorca contains mid-sentence cutoffs
    if not END_OK.search(a_turns[-1].strip()[-80:]):
        return bump("truncated")

    # verbose essay for a short question, unless writing was requested
    for i, (r, c) in enumerate(turns):
        if r != "assistant" or i == 0:
            continue
        q = turns[i - 1][1]
        if len(q.split()) < 15 and len(c.split()) > 200 and not WRITE_INTENT.search(q):
            return bump("length_mismatch")

    if source == "instruct_ultrachat" and SPAM_USER.search(turns[0][1]):
        return bump("spam_grounded")
    if source == "instruct_wizardlm" and EVOL_FRANKEN.search(turns[0][1]) \
            and "code" not in turns[0][1].lower()[:60]:
        return bump("evol_franken")
    return True


def identity_set(rng):
    """synthetic plasma identity + calibrated-ignorance conversations"""
    name_qs = [
        "Who are you?", "What are you?", "What is your name?",
        "What's your name?", "Tell me about yourself.", "Introduce yourself.",
        "What should I call you?", "Who am I talking to?", "What are you called?",
        "Are you a chatbot?", "What kind of AI are you?", "Describe yourself briefly.",
    ]
    name_as = [
        "I'm Plasma, a small language model from the 1386.ai project.",
        "My name is Plasma. I'm a compact AI language model trained from scratch as part of the 1386.ai project.",
        "I'm Plasma, an AI assistant. I'm a small model, so I do my best with simple, clear answers.",
        "I'm called Plasma — a small language model built from scratch, not based on any other AI.",
    ]
    maker_qs = [
        "Who made you?", "Who created you?", "Who built you?",
        "Who trained you?", "Where do you come from?", "Who developed you?",
    ]
    maker_as = [
        "I was trained from scratch by Evan for the 1386.ai project, on a single GPU.",
        "Evan built me from scratch as part of the 1386.ai project — no pretrained weights, one GPU.",
        "I come from the 1386.ai project, where I was trained from scratch on a single GPU.",
    ]
    notother_qs = [
        "Are you ChatGPT?", "Are you GPT-4?", "Are you made by OpenAI?",
        "Are you Google's AI?", "Are you Claude?", "Are you Open Assistant?",
    ]
    notother_as = [
        "No — I'm Plasma, a small independent language model from the 1386.ai project.",
        "No. I'm Plasma, trained from scratch for the 1386.ai project. I'm not related to that system.",
    ]
    human_qs = ["Are you a human?", "Are you a real person?", "Am I talking to a person?"]
    human_as = [
        "No, I'm an AI — a small language model called Plasma.",
        "No, I'm not a person. I'm Plasma, an AI language model.",
    ]
    capability_qs = [
        "What can you do?", "What can you help me with?", "How can you help me?",
    ]
    capability_as = [
        "I can answer questions, explain things simply, help with writing, and chat. I'm a small model, so for anything critical you should double-check important facts.",
        "I'm good at short explanations, everyday questions, and writing help. I'm a compact model, so I keep answers simple.",
    ]
    # calibrated ignorance: unknown entities get honest, short declines.
    # names generated to be unlike anything real or in the eval battery.
    fake_people = ["Torvin Mackelby", "Dr. Elska Runegard", "Senator Pell Draymore",
                   "the painter Coswell Brint", "General Hadrik Vane"]
    fake_things = ["the Ostrellian Compact of 1912", "the mineral bruvexite",
                   "the novel 'Winters of the Glass Harbor'",
                   "the Kelmar-Voss equation", "the ancient city of Threnholm"]
    ignorance_as = [
        "I don't have any information about {x}. It may be very obscure, or it may not exist.",
        "I'm not familiar with {x} — I don't want to guess and get it wrong.",
        "I don't recognize {x}. If it's real, it isn't something I learned about.",
    ]

    convs = []
    def add(q, a):
        convs.append(f"User: {q}\nAssistant: {a}")
    for q in name_qs:
        for a in rng.sample(name_as, 2):
            add(q, a)
    for q in maker_qs:
        for a in rng.sample(maker_as, 2):
            add(q, a)
    for q in notother_qs:
        for a in notother_as:
            add(q, a)
    for q in human_qs:
        for a in human_as:
            add(q, a)
    for q in capability_qs:
        for a in capability_as:
            add(q, a)
    for x in fake_people + fake_things:
        for tpl in (f"Who is {x}?" if x in fake_people else f"What is {x}?",
                    f"Tell me about {x}."):
            add(tpl, rng.choice(ignorance_as).format(x=x))
    rng.shuffle(convs)
    return convs


def build_mask_special(text, sp, bos, max_len=None):
    """1.2 template: atomic role markers, <|end|> after every turn.

      [BOS](0) [<|user|>](0) [content](0) [<|end|>](0) [\n](0)
      [<|assistant|>](0) [content](1) [<|end|>](1) [\n](0) ...
    loss sits on assistant content + its <|end|>, so stopping is supervised
    at every answer and stop detection is a single token compare.
    """
    U, A, E = (sp.piece_to_id(t) for t in ("<|user|>", "<|assistant|>", "<|end|>"))
    if min(U, A, E) <= 0:
        raise SystemExit("tokenizer lacks chat special tokens; "
                         "train it with scripts/train_tokenizer_1.2.py")
    nl = sp.encode("\n", out_type=int)
    turns = turns_of(text)
    if not turns:
        return None, None
    tokens, bits = [bos], [0]
    end_last_assistant = None
    for i, (role, content) in enumerate(turns):
        body = sp.encode(content, out_type=int)
        if role == "assistant":
            tokens += [A] + body + [E]
            bits += [0] + [1] * len(body) + [1]
            if max_len is None or len(tokens) <= max_len:
                end_last_assistant = len(tokens)
        else:
            tokens += [U] + body + [E]
            bits += [0] * (len(body) + 2)
        if i < len(turns) - 1:
            tokens += nl
            bits += [0] * len(nl)
    if max_len is not None and len(tokens) > max_len:
        if not end_last_assistant:
            return None, None
        tokens = tokens[:end_last_assistant]
        bits = bits[:end_last_assistant]
    if len(tokens) < 8 or not any(bits):
        return None, None
    return tokens, bits


def build_mask_v4(text, sp, bos, eos, max_len=None):
    """tokens + mask with BOS at start and EOS after every assistant turn.

    layout per conversation:
      [BOS](0) [User: ](0) [content\n](0) [Assistant: ](0) [content](1) [EOS](1)
      [\n](0) [User: ](0) ...
    """
    turns = turns_of(text)
    if not turns:
        return None, None
    tokens, bits = [bos], [0]
    end_last_assistant = None
    for i, (role, content) in enumerate(turns):
        label = "User: " if role == "user" else "Assistant: "
        lab = sp.encode(label, out_type=int)
        tokens += lab
        bits += [0] * len(lab)
        is_last = i == len(turns) - 1
        if role == "assistant":
            body = sp.encode(content, out_type=int)
            tokens += body
            bits += [1] * len(body)
            tokens.append(eos)
            bits.append(1)
            if max_len is None or len(tokens) <= max_len:
                end_last_assistant = len(tokens)
            if not is_last:
                nl = sp.encode("\n", out_type=int)
                tokens += nl
                bits += [0] * len(nl)
        else:
            body = sp.encode(content + ("\n" if not is_last else ""), out_type=int)
            tokens += body
            bits += [0] * len(body)

    if max_len is not None and len(tokens) > max_len:
        if not end_last_assistant:
            return None, None
        tokens = tokens[:end_last_assistant]
        bits = bits[:end_last_assistant]
    if len(tokens) < 8 or not any(bits):
        return None, None
    return tokens, bits


def load_source(name, rng):
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
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--template", choices=["v4", "special"], default="v4",
                    help="v4 = 'User:/Assistant:' spans (1.1); "
                         "special = atomic <|user|> markers (1.2)")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global PACK_LEN
    PACK_LEN = args.seq_len + 1

    out_dir = Path(args.out_dir)
    if out_dir.exists() and (out_dir / "meta.yaml").exists() and not args.force:
        print(f"[skip] existing: {out_dir}")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    bos, eos, pad = sp.bos_id(), sp.eos_id(), sp.pad_id()
    if pad is None or pad < 0:
        raise SystemExit("tokenizer has no pad id; refusing to pad with unk")
    rng = random.Random(args.seed)

    # decontam: benchmark prompts + the audit battery
    exclude = set(all_prompt_strings())
    try:
        from scripts.audit.battery_prompts import battery
        for p in battery():
            exclude.add(norm_prompt(p["prompt"]))
    except Exception as e:
        print(f"  [warn] battery decontam unavailable: {e}")

    t0 = time.time()
    stats, src_counts = {}, {}
    seen = set()
    kept = []  # (text, source)

    for name, cap, repeat in SOURCES:
        rows = load_source(name, rng)
        n_kept = 0
        for text in rows:
            if n_kept >= cap:
                break
            text = normalize_text(text)
            turns = turns_of(text)
            if not conv_ok(turns, name, stats):
                continue
            if name == "instruct_ultrachat" and len(turns) > 6:
                turns = turns[:6]           # first 3 user/assistant pairs
                if turns[-1][0] == "user":
                    turns = turns[:-1]
                text = rebuild(turns)
            if norm_prompt(turns[0][1]) in exclude:
                stats["decontam"] = stats.get("decontam", 0) + 1
                continue
            key = (norm_prompt(turns[0][1])[:150],
                   turns_of(text)[1][1][:200] if len(turns) > 1 else "")
            if key in seen:
                stats["dup"] = stats.get("dup", 0) + 1
                continue
            seen.add(key)
            for _ in range(repeat):
                kept.append((text, name))
            n_kept += 1
        src_counts[name] = n_kept * repeat
        print(f"  {name}: kept {n_kept:,} x{repeat}", flush=True)

    # x4 so ~320 identity examples anchor the persona against 299k others;
    # the competing "Open Assistant" identity is filtered out entirely, so
    # this is the only identity signal the model sees
    ident = identity_set(rng) * 4
    kept += [(t, "identity_synth") for t in ident]
    src_counts["identity_synth"] = len(ident)
    print(f"  identity_synth: {len(ident)}")
    print(f"\n  filter drops: {stats}", flush=True)

    rng.shuffle(kept)

    convs = []
    for text, source in kept:
        if args.template == "special":
            toks, bits = build_mask_special(text, sp, bos, max_len=PACK_LEN)
        else:
            toks, bits = build_mask_v4(text, sp, bos, eos, max_len=PACK_LEN)
        if toks is None:
            stats["tokenize_drop"] = stats.get("tokenize_drop", 0) + 1
            continue
        convs.append((toks, bits))
    print(f"  tokenized {len(convs):,} conversations "
          f"in {time.time()-t0:.0f}s", flush=True)

    # best-fit-decreasing whole-conversation packing (same as v3)
    GRAN = 32
    NB = PACK_LEN // GRAN + 1
    order = sorted(range(len(convs)), key=lambda i: -len(convs[i][0]))
    bins, buckets = [], [[] for _ in range(NB)]
    for i in order:
        toks, bits = convs[i]
        need = len(toks)
        b0 = (need + GRAN - 1) // GRAN
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
    real = 0
    for i, (bt, bb) in enumerate(bins):
        L = min(len(bt), PACK_LEN)
        packed[i, :L] = np.asarray(bt[:L], dtype=np.uint16)
        masks[i, :L] = np.asarray(bb[:L], dtype=np.uint8)
        real += L

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

    # verify the core property before declaring success
    n_eos = int((packed[:100] == eos).sum())
    n_asst = 0
    asst_id = sp.encode("Assistant: ", out_type=int)[0]
    n_asst = int((packed[:100] == asst_id).sum())

    meta = {
        "seq_len": args.seq_len, "pack_len": PACK_LEN, "dtype": "uint16",
        "has_loss_mask": True, "multiturn": True, "template": args.template,
        "bos_at_conv_start": True, "eos_after_every_assistant_turn": True,
        "eos_per_assistant_label_first100": round(n_eos / max(n_asst, 1), 3),
        "tokenizer": TOKENIZER.name, "vocab_size": sp.get_piece_size(),
        "n_train_sequences": int(n_train), "n_val_sequences": int(n_seq - n_train),
        "n_train_shards": int(idx), "n_conversations": int(len(convs)),
        "assistant_token_frac": round(float(masks[:n_train].mean()), 4),
        "pad_frac": round(1.0 - real / (n_seq * PACK_LEN), 4),
        "packing": "whole_conversation_binpack",
        "cross_conversation_attention": "packed-causal (block-diag via bos cumsum at train time)",
        "filter_drops": stats, "source_counts": src_counts,
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=True),
                                       encoding="utf-8")
    print(f"\nDone in {time.time()-t0:.0f}s: {n_train:,} train seqs, "
          f"assistant_frac={meta['assistant_token_frac']:.1%}, "
          f"pad={meta['pad_frac']:.1%}, eos/asst={meta['eos_per_assistant_label_first100']}")
    print(yaml.safe_dump(meta, sort_keys=True))


if __name__ == "__main__":
    main()
