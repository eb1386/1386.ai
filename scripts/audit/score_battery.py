#!/usr/bin/env python3
# battery scorer
#
# reads runner jsonl, applies each prompt's check, adds audit flags
# (hedging, fake dialogue, identity leaks, mojibake, verbosity), and prints
# per-condition and per-category tables. writes a scored jsonl beside the input.
#
#   python scripts/audit/score_battery.py logs/audit/grid.jsonl logs/audit/grid_s1.jsonl

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_v2 import run_code  # noqa: E402

HEDGES = ("i'm not sure", "i am not sure", "i think", "i believe",
          "it's possible", "as far as i know", "i don't know", "i'm not certain")
IDENTITY = ("open assistant", "chatgpt", "openai", "anthropic", "i am gpt", "google assistant")
SPEAKER = re.compile(r"(?:^|\n)\s*(?:User|Assistant|System|Human|Q|A|B|Boy|Girl|Teacher|Student)\s?:", re.I)
NUMBER_CLEAN = re.compile(r"(?<=\d),(?=\d)")
ITEM_LINE = re.compile(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)(.+)$")


def norm(s):
    return s.lower().strip()


def count_items(resp):
    lines = [m.group(1) for m in (ITEM_LINE.match(l) for l in resp.split("\n")) if m]
    if len(lines) >= 2:
        return len(lines)
    # fall back to comma/and separation on the first sentence
    first = re.split(r"[.!?\n]", resp)[0]
    first = re.sub(r"\band\b", ",", first, flags=re.I)
    if ":" in first:
        first = first.split(":", 1)[1]
    parts = [p.strip() for p in first.split(",") if p.strip()]
    return len(parts)


def check_ok(rec):
    kind, expect, resp = rec["check"], rec["expect"], rec["output"]
    low = norm(resp)
    if kind == "none" or not resp:
        return None
    if kind == "contains":
        return int(any(norm(k) in low for k in expect))
    if kind == "number":
        cleaned = NUMBER_CLEAN.sub("", resp)
        return int(re.search(rf"(?<![\d.]){re.escape(expect)}(?![\d])", cleaned) is not None)
    if kind == "count":
        n_ok = count_items(resp) == expect["n"]
        kw_ok = all(norm(k) in low for k in expect["kw"])
        return int(n_ok and kw_ok)
    if kind == "word_limit":
        return int(len(resp.split()) <= expect)
    if kind == "yesno":
        first = re.sub(r"[^a-z]", "", low.split()[0]) if low.split() else ""
        return int(first == expect)
    if kind == "multi":
        return int(all(not kws or any(norm(k) in low for k in kws)
                       for kws in expect["parts"]))
    if kind == "exec":
        try:
            return int(bool(run_code(resp, expect["fn"],
                                     [(tuple(a), e) for a, e in expect["tests"]])))
        except Exception:
            return 0
    return None


def flags(rec):
    resp, low = rec["output"], norm(rec["output"])
    return {
        "hedge": int(any(low.startswith(h) for h in HEDGES)),
        "identity": int(any(i in low for i in IDENTITY)),
        "fake_dialogue": int(bool(SPEAKER.search(resp))),
        "mojibake": int(("â€" in resp) or ("�" in resp) or ("â€" in resp)),
        "verbose": int(rec["n_new"] > 120 and rec["check"] in
                       ("contains", "number", "yesno", "count")),
    }


def main(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    print(f"loaded {len(rows)} generations from {len(paths)} file(s)\n")

    scored = []
    for r in rows:
        r["ok"] = check_ok(r)
        r.update(flags(r))
        scored.append(r)

    out = Path(paths[0]).with_suffix(".scored.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    conds = sorted({r["condition"] for r in scored})
    print(f"{'condition':>10} {'acc':>6} {'n_scored':>9} {'eos%':>6} "
          f"{'mean_tok':>9} {'hedge%':>7} {'fakedlg%':>9} {'ident%':>7} "
          f"{'moji%':>6} {'verbose%':>9}")
    for c in conds:
        cr = [r for r in scored if r["condition"] == c]
        ck = [r for r in cr if r["ok"] is not None]
        acc = sum(r["ok"] for r in ck) / max(1, len(ck))
        eos = sum(r["stop"] == "eos" for r in cr) / len(cr)
        toks = sum(r["n_new"] for r in cr) / len(cr)
        pct = lambda k: 100 * sum(r[k] for r in cr) / len(cr)
        print(f"{c:>10} {acc:>6.3f} {len(ck):>9} {100*eos:>5.0f}% "
              f"{toks:>9.0f} {pct('hedge'):>6.1f}% {pct('fake_dialogue'):>8.1f}% "
              f"{pct('identity'):>6.1f}% {pct('mojibake'):>5.1f}% {pct('verbose'):>8.1f}%")

    print("\nper-category accuracy by condition (checkable prompts only)")
    cats = sorted({r["category"] for r in scored if r["ok"] is not None})
    hdr = f"{'category':>13} " + " ".join(f"{c:>9}" for c in conds)
    print(hdr)
    for cat in cats:
        cells = []
        for c in conds:
            cr = [r for r in scored
                  if r["category"] == cat and r["condition"] == c and r["ok"] is not None]
            cells.append(f"{sum(r['ok'] for r in cr)/len(cr):>9.2f}" if cr else f"{'-':>9}")
        print(f"{cat:>13} " + " ".join(cells))

    return scored


if __name__ == "__main__":
    main(sys.argv[1:])
