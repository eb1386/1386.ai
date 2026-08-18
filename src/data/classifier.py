# sklearn quality classifier

import os
import sys
import time
import pickle
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def train_quality_classifier(output_path: str, n_samples: int = 200_000,
                             seed: int = 42):
    """Train quality classifier using local data files."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier

    print(f"Training quality classifier ({n_samples:,} samples)...")
    t0 = time.time()

    raw_dir = ROOT / "data" / "raw_1.1"
    rng = random.Random(seed)
    half = n_samples // 2

    # high quality: fineweb-edu (already filtered for quality)
    # low quality: random short/repetitive docs from any source
    texts = []
    labels = []

    # collect high quality from fineweb-edu
    fineweb = raw_dir / "fineweb_edu_hq.txt"
    if fineweb.exists():
        print("  Loading high-quality samples from fineweb-edu...")
        doc = []
        high = 0
        with open(fineweb, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "" and doc:
                    text = " ".join(doc)[:2000]
                    if len(text) > 200 and rng.random() < 0.01:
                        texts.append(text)
                        labels.append(1)
                        high += 1
                        if high >= half:
                            break
                    doc = []
                else:
                    doc.append(line.strip())
        print(f"  High quality: {high:,}")

    # generate low quality by corrupting/truncating text
    print("  Generating low-quality samples...")
    low = 0
    low_sources = []

    for src_file in raw_dir.glob("*.txt"):
        with open(src_file, "r", encoding="utf-8") as f:
            doc = []
            for line in f:
                if line.strip() == "" and doc:
                    text = " ".join(doc)
                    # short, repetitive, or truncated = low quality
                    if len(text) < 200 and rng.random() < 0.05:
                        low_sources.append(text[:500])
                    elif rng.random() < 0.005:
                        # truncate randomly
                        low_sources.append(text[:50])
                    elif rng.random() < 0.003:
                        # repeat a phrase
                        words = text.split()[:10]
                        low_sources.append(" ".join(words * 5))
                    doc = []
                    if len(low_sources) >= half:
                        break
                else:
                    doc.append(line.strip())
        if len(low_sources) >= half:
            break

    rng.shuffle(low_sources)
    for text in low_sources[:half]:
        texts.append(text)
        labels.append(0)
        low += 1

    print(f"  Low quality: {low:,}")
    print(f"  Total: {len(texts):,} samples")

    if len(texts) < 100 or low == 0 or len([l for l in labels if l == 1]) == 0:
        raise ValueError("Not enough data for both classes")

    print(f"  Training classifier...")
    vectorizer = HashingVectorizer(
        n_features=2**18, ngram_range=(1, 2),
        alternate_sign=False, dtype="float32",
    )
    X = vectorizer.transform(texts)

    clf = SGDClassifier(loss="log_loss", random_state=seed, n_jobs=-1)
    clf.fit(X, labels)

    model = {"vectorizer": vectorizer, "classifier": clf}
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    dt = time.time() - t0
    print(f"  Quality classifier saved: {output_path} ({dt:.0f}s)")
    return model


def train_toxicity_classifier(output_path: str, n_samples: int = 200_000,
                              seed: int = 42):
    """Train toxicity classifier using civil comments data."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Training toxicity classifier ({n_samples:,} samples)...")
    t0 = time.time()

    ds = load_dataset("google/civil_comments", split="train",
                      streaming=True)

    texts = []
    labels = []
    toxic = 0
    clean = 0
    half = n_samples // 2

    for example in ds:
        score = example.get("toxicity", None)
        if score is None:
            continue

        text = example.get("text", "").strip().replace("\n", " ")[:1000]
        if len(text) < 20:
            continue

        if score >= 0.5 and toxic < half:
            texts.append(text)
            labels.append(1)
            toxic += 1
        elif score < 0.1 and clean < half:
            texts.append(text)
            labels.append(0)
            clean += 1

        if toxic >= half and clean >= half:
            break

        if (toxic + clean) % 50000 == 0 and (toxic + clean) > 0:
            print(f"  {toxic + clean:,} samples (toxic: {toxic:,}, clean: {clean:,})")

    print(f"  Collected {toxic + clean:,} samples (toxic: {toxic:,}, clean: {clean:,})")
    print(f"  Training classifier...")

    vectorizer = HashingVectorizer(
        n_features=2**17, ngram_range=(1, 2),
        alternate_sign=False, dtype="float32",
    )
    X = vectorizer.transform(texts)

    clf = SGDClassifier(loss="log_loss", random_state=seed, n_jobs=-1)
    clf.fit(X, labels)

    model = {"vectorizer": vectorizer, "classifier": clf}
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    dt = time.time() - t0
    print(f"  Toxicity classifier saved: {output_path} ({dt:.0f}s)")
    return model


class QualityClassifier:
    """Score documents using trained classifier."""

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]

    def score(self, text: str) -> float:
        """Return 0-1 quality score."""
        text = text.strip().replace("\n", " ")[:2000]
        if not text:
            return 0.0
        X = self.vectorizer.transform([text])
        prob = self.classifier.predict_proba(X)[0]
        return float(prob[1]) if len(prob) > 1 else float(prob[0])

    def is_high_quality(self, text: str, threshold: float = 0.6) -> bool:
        return self.score(text) >= threshold


class ToxicityClassifier:
    """Filter toxic documents using trained classifier."""

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]

    def toxicity_score(self, text: str) -> float:
        """Return 0-1 toxicity score."""
        text = text.strip().replace("\n", " ")[:1000]
        if not text:
            return 0.0
        X = self.vectorizer.transform([text])
        prob = self.classifier.predict_proba(X)[0]
        return float(prob[1]) if len(prob) > 1 else float(prob[0])

    def is_toxic(self, text: str, threshold: float = 0.5) -> bool:
        return self.toxicity_score(text) >= threshold
