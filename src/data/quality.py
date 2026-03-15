# document quality scoring

import re
import math
from collections import Counter

_STOPWORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "was", "are", "were", "been",
    "has", "had", "did", "does", "being", "am",
})

_BOILERPLATE_PATTERNS = [
    re.compile(r"cookie(?:s)?\s+(?:policy|consent|settings)", re.IGNORECASE),
    re.compile(r"subscribe\s+(?:to\s+)?(?:our\s+)?newsletter", re.IGNORECASE),
    re.compile(r"click\s+here\s+to\s+(?:read|learn|subscribe|sign)", re.IGNORECASE),
    re.compile(r"all\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"terms\s+(?:of\s+)?(?:service|use)|privacy\s+policy", re.IGNORECASE),
    re.compile(r"(?:share|follow)\s+(?:on|us\s+on)\s+(?:twitter|facebook|instagram|linkedin)", re.IGNORECASE),
    re.compile(r"(?:sign\s+up|log\s*in|create\s+(?:an?\s+)?account)", re.IGNORECASE),
    re.compile(r"(?:advertisement|sponsored\s+content|promoted)", re.IGNORECASE),
    re.compile(r"\u00a9\s*\d{4}", re.IGNORECASE),
    re.compile(r"loading\.\.\.|please\s+wait", re.IGNORECASE),
    re.compile(r"enable\s+javascript", re.IGNORECASE),
    re.compile(r"your\s+(?:email|e-mail)\s+address", re.IGNORECASE),
    re.compile(r"unsubscribe\s+(?:from|at\s+any\s+time)", re.IGNORECASE),
]

_TEMPLATE_PATTERNS = [
    re.compile(r"\{[a-z_]+\}"),
    re.compile(r"lorem\s+ipsum", re.IGNORECASE),
    re.compile(r"(?:TODO|FIXME|XXX|HACK):", re.IGNORECASE),
]


def _word_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())


def _sentence_split(text: str) -> list[str]:
    sents = re.split(r"[.!?]+\s+", text)
    return [s.strip() for s in sents if len(s.strip()) > 5]


def score_document(text: str) -> dict:
    """Score document on multiple quality signals. Returns dict with 0-1 scores."""
    scores = {}
    words = _word_tokenize(text)
    n_words = len(words)

    if n_words < 20:
        return {"quality_score": 0.0, "reason": "too_short"}

    # length
    if n_words < 50:
        scores["length"] = 0.2
    elif n_words < 200:
        scores["length"] = 0.5 + 0.3 * (n_words - 50) / 150
    elif n_words <= 5000:
        scores["length"] = 1.0
    elif n_words <= 20000:
        scores["length"] = 1.0 - 0.3 * (n_words - 5000) / 15000
    else:
        scores["length"] = 0.5

    # diversity
    unique_words = len(set(words))
    corrected_ttr = unique_words / math.sqrt(n_words) if n_words > 0 else 0
    if corrected_ttr < 2:
        scores["diversity"] = 0.1
    elif corrected_ttr < 4:
        scores["diversity"] = 0.3 + 0.4 * (corrected_ttr - 2) / 2
    elif corrected_ttr <= 20:
        scores["diversity"] = 0.8 + 0.2 * min(1.0, (corrected_ttr - 4) / 8)
    else:
        scores["diversity"] = 0.9

    # sentence structure
    sentences = _sentence_split(text)
    n_sents = len(sentences)
    if n_sents < 2:
        scores["sentence_structure"] = 0.3
    else:
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths)
        sent_len_std = (sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths)) ** 0.5
        length_score = 1.0
        if avg_sent_len < 5:
            length_score = 0.3
        elif avg_sent_len < 10:
            length_score = 0.6
        elif avg_sent_len > 50:
            length_score = 0.5
        variety_score = min(1.0, sent_len_std / 8)
        scores["sentence_structure"] = 0.6 * length_score + 0.4 * variety_score

    # paragraph structure
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    n_paras = len(paragraphs)
    if n_paras == 0:
        scores["paragraph_structure"] = 0.2
    elif n_paras == 1:
        scores["paragraph_structure"] = 0.5 if n_words > 200 else 0.7
    else:
        para_lengths = [len(p.split()) for p in paragraphs]
        avg_para = sum(para_lengths) / len(para_lengths)
        if 30 <= avg_para <= 200:
            scores["paragraph_structure"] = 1.0
        elif avg_para < 10:
            scores["paragraph_structure"] = 0.4
        else:
            scores["paragraph_structure"] = 0.7

    # naturalness
    stopword_count = sum(1 for w in words if w in _STOPWORDS)
    stopword_ratio = stopword_count / n_words if n_words > 0 else 0
    if 0.30 <= stopword_ratio <= 0.60:
        scores["naturalness"] = 1.0
    elif 0.20 <= stopword_ratio < 0.30:
        scores["naturalness"] = 0.6
    elif stopword_ratio < 0.20:
        scores["naturalness"] = 0.3
    elif stopword_ratio <= 0.70:
        scores["naturalness"] = 0.7
    else:
        scores["naturalness"] = 0.3

    # capitalization
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
    if lines:
        starts_cap = sum(1 for l in lines if l[0].isupper() or l[0].isdigit())
        scores["capitalization"] = min(1.0, starts_cap / len(lines) + 0.2)
    else:
        scores["capitalization"] = 0.5

    # repetition
    if n_words >= 50:
        trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)
        if trigrams:
            top_count = trigram_counts.most_common(1)[0][1]
            repeat_ratio = top_count / len(trigrams)
            if repeat_ratio > 0.05:
                scores["repetition"] = 0.2
            elif repeat_ratio > 0.02:
                scores["repetition"] = 0.6
            else:
                scores["repetition"] = 1.0
        else:
            scores["repetition"] = 0.5
    else:
        scores["repetition"] = 0.7

    # boilerplate
    boilerplate_hits = sum(1 for p in _BOILERPLATE_PATTERNS if p.search(text))
    if boilerplate_hits == 0:
        scores["boilerplate"] = 1.0
    elif boilerplate_hits <= 2:
        scores["boilerplate"] = 0.7
    elif boilerplate_hits <= 4:
        scores["boilerplate"] = 0.4
    else:
        scores["boilerplate"] = 0.1

    # template detection
    template_hits = sum(1 for p in _TEMPLATE_PATTERNS if p.search(text))
    scores["not_template"] = 1.0 if template_hits == 0 else max(0.2, 1.0 - template_hits * 0.3)

    # url density
    url_count = len(re.findall(r"https?://", text))
    email_count = len(re.findall(r"\S+@\S+\.\S+", text))
    special_density = (url_count + email_count) / max(1, n_words) * 100
    if special_density < 0.5:
        scores["clean_content"] = 1.0
    elif special_density < 2:
        scores["clean_content"] = 0.7
    elif special_density < 5:
        scores["clean_content"] = 0.4
    else:
        scores["clean_content"] = 0.1

    # weighted combine
    weights = {
        "length": 0.05, "diversity": 0.15, "sentence_structure": 0.15,
        "paragraph_structure": 0.05, "naturalness": 0.15, "capitalization": 0.05,
        "repetition": 0.15, "boilerplate": 0.10, "not_template": 0.05,
        "clean_content": 0.10,
    }
    scores["quality_score"] = round(sum(scores[k] * weights[k] for k in weights), 4)
    return scores


def filter_and_score(text: str, min_score: float = 0.5) -> tuple[bool, float]:
    """Quick pass/fail with score."""
    if len(text) < 100:
        return False, 0.0
    words = text.split()
    if len(words) < 20:
        return False, 0.0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / max(1, len(text)) < 0.80:
        return False, 0.0
    result = score_document(text)
    score = result["quality_score"]
    return score >= min_score, score


def score_batch(texts: list[str], min_score: float = 0.5) -> list[tuple[str, float]]:
    """Score batch, return passing docs sorted by score."""
    results = []
    for text in texts:
        passed, score = filter_and_score(text, min_score)
        if passed:
            results.append((text, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
