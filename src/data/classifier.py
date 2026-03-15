# fasttext quality classifier

import os
import sys
import tempfile
import time
from pathlib import Path


def train_quality_classifier(output_path: str, n_samples: int = 500_000,
                             seed: int = 42):
    """Train fasttext quality classifier using FineWeb-Edu scores as labels.

    FineWeb-Edu comes with a 0-5 quality score per document. We use
    those scores as ground truth to train a fast classifier that can
    then score ANY text, not just FineWeb documents.

    Labels:
      score >= 3.0  ->  __label__high
      score < 1.5   ->  __label__low
      1.5-3.0       ->  skipped (ambiguous)
    """
    import fasttext

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Training quality classifier ({n_samples:,} samples)...")
    t0 = time.time()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        train_path = f.name
        count = 0
        high = 0
        low = 0

        for example in ds:
            score = example.get("score", None)
            if score is None:
                continue

            text = example["text"].strip().replace("\n", " ").replace("\r", " ")
            if len(text) < 100:
                continue

            # truncate for training efficiency
            text = text[:2000]

            if score >= 3.0:
                f.write(f"__label__high {text}\n")
                high += 1
                count += 1
            elif score < 1.5:
                f.write(f"__label__low {text}\n")
                low += 1
                count += 1

            if count >= n_samples:
                break

            if count % 100000 == 0 and count > 0:
                print(f"  {count:,} samples (high: {high:,}, low: {low:,})")

    print(f"  Collected {count:,} samples (high: {high:,}, low: {low:,})")
    print(f"  Training fasttext...")

    model = fasttext.train_supervised(
        input=train_path,
        lr=0.5,
        epoch=5,
        wordNgrams=2,
        dim=100,
        loss="softmax",
        thread=os.cpu_count() or 4,
        verbose=0,
    )

    model.save_model(output_path)
    os.unlink(train_path)

    dt = time.time() - t0
    print(f"  Quality classifier saved: {output_path} ({dt:.0f}s)")
    return model


def train_toxicity_classifier(output_path: str, n_samples: int = 300_000,
                              seed: int = 42):
    """Train fasttext toxicity classifier using public toxic comment data.

    Uses the Jigsaw/Civil Comments dataset from HuggingFace. Labels
    documents as toxic or clean based on human annotations.
    """
    import fasttext

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Training toxicity classifier ({n_samples:,} samples)...")
    t0 = time.time()

    # civil comments has toxicity scores 0-1
    ds = load_dataset("google/civil_comments", split="train",
                      streaming=True, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        train_path = f.name
        count = 0
        toxic = 0
        clean = 0

        for example in ds:
            score = example.get("toxicity", None)
            if score is None:
                continue

            text = example.get("text", "").strip().replace("\n", " ").replace("\r", " ")
            if len(text) < 20:
                continue

            text = text[:1000]

            if score >= 0.5:
                f.write(f"__label__toxic {text}\n")
                toxic += 1
                count += 1
            elif score < 0.1:
                f.write(f"__label__clean {text}\n")
                clean += 1
                count += 1

            if count >= n_samples:
                break

            if count % 100000 == 0 and count > 0:
                print(f"  {count:,} samples (toxic: {toxic:,}, clean: {clean:,})")

    print(f"  Collected {count:,} samples (toxic: {toxic:,}, clean: {clean:,})")
    print(f"  Training fasttext...")

    model = fasttext.train_supervised(
        input=train_path,
        lr=0.5,
        epoch=5,
        wordNgrams=2,
        dim=50,
        loss="softmax",
        thread=os.cpu_count() or 4,
        verbose=0,
    )

    model.save_model(output_path)
    os.unlink(train_path)

    dt = time.time() - t0
    print(f"  Toxicity classifier saved: {output_path} ({dt:.0f}s)")
    return model


class QualityClassifier:
    """Score documents using trained fasttext classifier."""

    def __init__(self, model_path: str):
        import fasttext
        # suppress fasttext warnings about deprecated load_model
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        self.model = fasttext.load_model(model_path)

    def score(self, text: str) -> float:
        """Return 0-1 quality score. Higher is better."""
        text = text.strip().replace("\n", " ").replace("\r", " ")[:2000]
        if not text:
            return 0.0
        labels, probs = self.model.predict(text)
        label = labels[0]
        prob = probs[0]
        if label == "__label__high":
            return prob
        else:
            return 1.0 - prob

    def is_high_quality(self, text: str, threshold: float = 0.6) -> bool:
        return self.score(text) >= threshold


class ToxicityClassifier:
    """Filter toxic documents using trained fasttext classifier."""

    def __init__(self, model_path: str):
        import fasttext
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        self.model = fasttext.load_model(model_path)

    def toxicity_score(self, text: str) -> float:
        """Return 0-1 toxicity score. Higher is more toxic."""
        text = text.strip().replace("\n", " ").replace("\r", " ")[:1000]
        if not text:
            return 0.0
        labels, probs = self.model.predict(text)
        label = labels[0]
        prob = probs[0]
        if label == "__label__toxic":
            return prob
        else:
            return 1.0 - prob

    def is_toxic(self, text: str, threshold: float = 0.5) -> bool:
        return self.toxicity_score(text) >= threshold
