#!/usr/bin/env python3
# download free public instruction datasets for plasma 1.1 finetune
# writes one JSONL file per source into data/raw_1.1/
# format: {"text": "User: ...\nAssistant: ..."}

import argparse
import json
import sys
import time
from pathlib import Path

# windows console encoding fix
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "raw_1.1"


def banner(msg):
    print(f"\n{'-' * 60}\n  {msg}\n{'-' * 60}")


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def fmt_single(user_prompt: str, assistant_response: str) -> str:
    return f"User: {user_prompt.strip()}\nAssistant: {assistant_response.strip()}"


def fmt_multi(turns: list[tuple[str, str]]) -> str:
    """turns is a list of (role, content) where role in {user, assistant}."""
    parts = []
    for role, content in turns:
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {content.strip()}")
    return "\n".join(parts)


# ── 1. dolly-15k ─────────────────────────────────────────────────────
def download_dolly(load_dataset, target: int = 15000):
    out = OUT_DIR / "instruct_dolly.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("Dolly-15k (Databricks, human-written)")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    records = []
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        ctx = (ex.get("context") or "").strip()
        resp = (ex.get("response") or "").strip()
        if not instr or not resp:
            continue
        prompt = f"{instr}\n\n{ctx}" if ctx else instr
        if len(resp) < 5:
            continue
        records.append({"text": fmt_single(prompt, resp), "source": "dolly"})
        if len(records) >= target:
            break
    n = write_jsonl(out, records)
    print(f"  wrote {n:,} → {out.name}")


# ── 2. no_robots (HF H4) ─────────────────────────────────────────────
def download_no_robots(load_dataset, target: int = 10000):
    out = OUT_DIR / "instruct_no_robots.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("No Robots (10k human-written, HuggingFaceH4)")
    try:
        ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    records = []
    for ex in ds:
        msgs = ex.get("messages") or []
        turns = []
        for m in msgs:
            role = m.get("role", "")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role in ("user", "assistant"):
                turns.append((role, content))
        if len(turns) < 2:
            continue
        if turns[-1][0] != "assistant":
            continue
        records.append({"text": fmt_multi(turns), "source": "no_robots"})
        if len(records) >= target:
            break
    n = write_jsonl(out, records)
    print(f"  wrote {n:,} → {out.name}")


# ── 3. OpenAssistant oasst1 ──────────────────────────────────────────
def download_oasst(load_dataset, target: int = 60000):
    out = OUT_DIR / "instruct_oasst.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("OpenAssistant oasst1 (multi-turn, human-curated)")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # build message tree: id → message. only filter explicitly bad reviews.
    # do NOT filter on rank — most messages have rank=None and would be lost.
    by_id = {}
    children = {}
    for ex in ds:
        if (ex.get("lang") or "") != "en":
            continue
        if ex.get("deleted"):
            continue
        if ex.get("review_result") is False:
            continue
        mid = ex["message_id"]
        by_id[mid] = ex
        parent = ex.get("parent_id")
        children.setdefault(parent, []).append(mid)

    # walk root → leaf paths and emit each assistant terminus as a conversation.
    # explore ALL children (not just top-2) and dedup on full conversation hash
    # rather than first 200 chars (which collapsed near-identical variations).
    records = []
    seen = set()
    roots = children.get(None, [])

    def walk(path, depth=0):
        if not path or depth > 12:  # safety: cap recursion depth
            return
        last_id = path[-1]
        last = by_id.get(last_id)
        if not last:
            return
        if last["role"] == "assistant":
            turns = []
            for mid in path:
                msg = by_id[mid]
                role = "user" if msg["role"] == "prompter" else "assistant"
                content = (msg.get("text") or "").strip()
                if not content:
                    return
                turns.append((role, content))
            if len(turns) >= 2 and turns[-1][0] == "assistant":
                # hash the full conversation so different replies to the same
                # root prompt all count as distinct samples
                full = fmt_multi(turns)
                key = hash(full[:2000])
                if key not in seen:
                    seen.add(key)
                    records.append({"text": full, "source": "oasst1"})
        # walk all children, sorted by rank (best first), no limit
        kids = children.get(last_id, [])
        if not kids:
            return
        kids_sorted = sorted(
            kids,
            key=lambda k: by_id[k].get("rank") if by_id[k].get("rank") is not None else 99,
        )
        for k in kids_sorted:
            if len(records) >= target:
                return
            walk(path + [k], depth + 1)

    for r in roots:
        if len(records) >= target:
            break
        walk([r])

    n = write_jsonl(out, records[:target])
    print(f"  wrote {n:,} -> {out.name}")


# ── 4. SlimOrca (curated subset of OpenOrca) ─────────────────────────
def download_slimorca(load_dataset, target: int = 200000):
    out = OUT_DIR / "instruct_slimorca.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("SlimOrca (~500k → take 200k subset, GPT-4 reasoning)")
    try:
        ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    records = []
    for ex in ds:
        convs = ex.get("conversations") or []
        turns = []
        for m in convs:
            role_in = (m.get("from") or "").lower()
            content = (m.get("value") or "").strip()
            if not content:
                continue
            if role_in == "system":
                continue  # skip system prompts; we don't use them
            if role_in == "human":
                turns.append(("user", content))
            elif role_in == "gpt":
                turns.append(("assistant", content))
        if len(turns) < 2:
            continue
        if turns[-1][0] != "assistant":
            continue
        # filter overlong responses to keep packing efficient
        total_chars = sum(len(c) for _, c in turns)
        if total_chars > 6000:
            continue
        records.append({"text": fmt_multi(turns), "source": "slimorca"})
        if len(records) >= target:
            break
    n = write_jsonl(out, records)
    print(f"  wrote {n:,} → {out.name}")


# ── 5. UltraChat 200k subset ─────────────────────────────────────────
def download_ultrachat(load_dataset, target: int = 150000):
    out = OUT_DIR / "instruct_ultrachat.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("UltraChat 200k (multi-turn, take 150k subset)")
    try:
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    records = []
    for ex in ds:
        msgs = ex.get("messages") or []
        turns = []
        for m in msgs:
            role = m.get("role", "")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role in ("user", "assistant"):
                turns.append((role, content))
        if len(turns) < 2 or turns[-1][0] != "assistant":
            continue
        total_chars = sum(len(c) for _, c in turns)
        if total_chars > 6000:
            continue
        records.append({"text": fmt_multi(turns), "source": "ultrachat"})
        if len(records) >= target:
            break
    n = write_jsonl(out, records)
    print(f"  wrote {n:,} → {out.name}")


# ── 6. Alpaca (fallback if others fail) ──────────────────────────────
def download_alpaca(load_dataset, target: int = 50000):
    out = OUT_DIR / "instruct_alpaca.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("Alpaca (52k, classic instruction-tuning data)")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    records = []
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out_text = (ex.get("output") or "").strip()
        if not instr or not out_text or len(out_text) < 5:
            continue
        prompt = f"{instr}\n\n{inp}" if inp else instr
        records.append({"text": fmt_single(prompt, out_text), "source": "alpaca"})
        if len(records) >= target:
            break
    n = write_jsonl(out, records)
    print(f"  wrote {n:,} → {out.name}")


# ── 7. WizardLM evol-instruct (complex multi-step instructions) ──────
def download_wizardlm(load_dataset, target: int = 70000):
    out = OUT_DIR / "instruct_wizardlm.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("WizardLM evol-instruct (70k complex instructions)")
    try:
        ds = load_dataset(
            "WizardLMTeam/WizardLM_evol_instruct_70k", split="train"
        )
    except Exception as e:
        # alt mirror
        try:
            ds = load_dataset("cognitivecomputations/WizardLM_evol_instruct_70k_unfiltered",
                              split="train")
        except Exception as e2:
            print(f"  ERROR (both mirrors): {e}; {e2}")
            return

    records = []
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        out_text = (ex.get("output") or ex.get("response") or "").strip()
        if not instr or not out_text or len(out_text) < 5:
            continue
        if len(out_text) > 5000:
            continue
        records.append({"text": fmt_single(instr, out_text), "source": "wizardlm"})
        if len(records) >= target:
            break
    n = write_jsonl(out, records)
    print(f"  wrote {n:,} → {out.name}")


# ── 8. HH-RLHF helpful (Anthropic helpfulness data) ──────────────────
def download_hh_helpful(load_dataset, target: int = 50000):
    out = OUT_DIR / "instruct_hh_helpful.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("HH-RLHF helpful (Anthropic, ~50k helpful conversations)")
    try:
        ds = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="helpful-base",
            split="train",
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    import re
    records = []
    for ex in ds:
        chosen = (ex.get("chosen") or "").strip()
        if not chosen:
            continue
        # parse "\n\nHuman: ... \n\nAssistant: ..." into turns
        parts = re.split(r"\n\n(Human|Assistant): ", "\n\n" + chosen)
        # parts will be ['', 'Human', '<text>', 'Assistant', '<text>', ...]
        turns = []
        i = 1
        while i + 1 < len(parts):
            role_in = parts[i]
            content = parts[i + 1].strip()
            if content:
                role = "user" if role_in == "Human" else "assistant"
                turns.append((role, content))
            i += 2
        if len(turns) < 2 or turns[-1][0] != "assistant":
            continue
        total_chars = sum(len(c) for _, c in turns)
        if total_chars > 5000:
            continue
        records.append({"text": fmt_multi(turns), "source": "hh_helpful"})
        if len(records) >= target:
            break
    n = write_jsonl(out, records)
    print(f"  wrote {n:,} → {out.name}")


# ── 9. MetaMathQA (math reasoning) ───────────────────────────────────
def download_metamath(load_dataset, target: int = 50000):
    out = OUT_DIR / "instruct_metamath.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("MetaMathQA (math reasoning, 50k subset)")
    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    records = []
    for ex in ds:
        q = (ex.get("query") or ex.get("original_question") or "").strip()
        a = (ex.get("response") or "").strip()
        if not q or not a or len(a) < 10:
            continue
        if len(a) > 4000:
            continue
        records.append({"text": fmt_single(q, a), "source": "metamath"})
        if len(records) >= target:
            break
    n = write_jsonl(out, records)
    print(f"  wrote {n:,} -> {out.name}")


# ── 10. Capybara (high-quality multi-turn) ───────────────────────────
def download_capybara(load_dataset, target: int = 16000):
    out = OUT_DIR / "instruct_capybara.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name}")
        return
    banner("Capybara (16k high-quality multi-turn, distilled)")
    try:
        ds = load_dataset("LDJnr/Capybara", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    records = []
    for ex in ds:
        # capybara format: {"conversation": [{"input": "...", "output": "..."}, ...]}
        conv = ex.get("conversation") or []
        turns = []
        for t in conv:
            u = (t.get("input") or "").strip()
            a = (t.get("output") or "").strip()
            if u:
                turns.append(("user", u))
            if a:
                turns.append(("assistant", a))
        if len(turns) < 2 or turns[-1][0] != "assistant":
            continue
        total_chars = sum(len(c) for _, c in turns)
        if total_chars > 8000:
            continue
        records.append({"text": fmt_multi(turns), "source": "capybara"})
        if len(records) >= target:
            break
    n = write_jsonl(out, records)
    print(f"  wrote {n:,} -> {out.name}")


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download free public instruction datasets for Plasma 1.1"
    )
    parser.add_argument(
        "--sources", nargs="+",
        default=["dolly", "no_robots", "oasst", "slimorca", "ultrachat",
                 "alpaca", "wizardlm", "hh_helpful", "metamath", "capybara"],
        help="Which datasets to download",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    handlers = {
        "dolly": download_dolly,
        "no_robots": download_no_robots,
        "oasst": download_oasst,
        "slimorca": download_slimorca,
        "ultrachat": download_ultrachat,
        "alpaca": download_alpaca,
        "wizardlm": download_wizardlm,
        "hh_helpful": download_hh_helpful,
        "metamath": download_metamath,
        "capybara": download_capybara,
    }

    t0 = time.time()
    for name in args.sources:
        if name not in handlers:
            print(f"  unknown source: {name}")
            continue
        try:
            handlers[name](load_dataset)
        except Exception as e:
            print(f"  {name} failed: {e}")

    print(f"\n{'-' * 60}\nDownload summary:")
    total = 0
    for f in sorted(OUT_DIR.glob("instruct_*.jsonl")):
        n = sum(1 for _ in open(f, encoding="utf-8"))
        total += n
        print(f"  {f.name:35s} {n:>8,} samples  {f.stat().st_size / 1e6:>6.1f} MB")
    if (OUT_DIR / "synthetic_instruct.jsonl").exists():
        n = sum(1 for _ in open(OUT_DIR / "synthetic_instruct.jsonl", encoding="utf-8"))
        total += n
        print(f"  {'synthetic_instruct.jsonl':35s} {n:>8,} samples")
    print(f"  {'TOTAL':35s} {total:>8,} samples")
    print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
