# chat template at token level
#
# sft shards were built by span-encoding each piece separately
# (build_sft_v3.build_mask), so the role header "Assistant: " is
# ['▁Assistant', ':'] — ids a full-string encode can never produce after a
# newline. every inference path must therefore assemble prompts from the
# same spans, or the model starts generation from tokens it never trained on.

def _enc(sp, text):
    return sp.encode(text, out_type=int)


def label_ids(sp, role):
    # "User: " -> ['▁User', ':'], "Assistant: " -> ['▁Assistant', ':']
    return _enc(sp, f"{role}: ")


def turn_ids(sp, role, content, trailing_newline=True):
    """one 'Role: content\\n' turn, encoded the way training encoded it."""
    text = content + ("\n" if trailing_newline else "")
    return label_ids(sp, role) + _enc(sp, text)


def build_prompt_ids(sp, message, history=None, max_prompt_tokens=None,
                     eos_between_turns=False, with_bos=False):
    """full prompt ids ending in the open 'Assistant: ' header.

    eos_between_turns + with_bos match v4-style data (bos at conversation
    start, eos after every assistant answer); v3 data had neither inside a
    conversation, so keep both False for v3 checkpoints.
    """
    eos = sp.eos_id()
    chunks = []  # list of (role, ids) so old pairs can be dropped whole
    for m in history or []:
        role = "User" if m.get("role") == "user" else "Assistant"
        ids = turn_ids(sp, role, m.get("content", ""))
        if eos_between_turns and role == "Assistant":
            # eos sits before the newline, exactly as v4 shards place it
            ids = turn_ids(sp, role, m.get("content", ""), trailing_newline=False)
            ids = ids + [eos] + _enc(sp, "\n")
        chunks.append((role, ids))
    chunks.append(("User", turn_ids(sp, "User", message)))
    tail = label_ids(sp, "Assistant")

    if max_prompt_tokens:
        # drop oldest user+assistant pairs until the prompt fits
        def total():
            return sum(len(c[1]) for c in chunks) + len(tail)
        while len(chunks) > 1 and total() > max_prompt_tokens:
            chunks.pop(0)
            if chunks and chunks[0][0] == "Assistant":
                chunks.pop(0)

    ids = [sp.bos_id()] if with_bos else []
    for _, c in chunks:
        ids.extend(c)
    ids.extend(tail)
    return ids


def stop_sequences(sp):
    """generated-id suffixes that mean a fake next turn is starting."""
    nl = _enc(sp, "a\nb")  # recover the newline byte token from context
    nl_id = [t for t in nl if sp.id_to_piece(t) == "<0x0A>"]
    if not nl_id:
        return []
    n = nl_id[0]
    user = label_ids(sp, "User")
    asst = label_ids(sp, "Assistant")
    # require the ':' so a line starting with the bare word (e.g.
    # "Usernames should...") is never mistaken for a role header
    return [[n] + user, [n] + asst]


def penalty_exclude(sp):
    """structure tokens the repetition penalty must never suppress.

    digits are here because split-digit tokenization makes every number a
    digit sequence: penalizing repeats turned '85 + 34 = 119' impossible
    (measured: math 0.00 at rp 1.3 vs 0.62 unpenalized).
    """
    ids = set()
    for piece in (".", ",", ":", ";", "?", "!", "'", '"', ")", "(",
                  "▁", "▁the", "▁a", "▁of", "▁to", "▁is", "▁and",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
        pid = sp.piece_to_id(piece)
        if pid > 0:
            ids.add(pid)
    for t in _enc(sp, "a\nb"):
        if sp.id_to_piece(t) == "<0x0A>":
            ids.add(t)
    ids.add(sp.eos_id())
    return ids
