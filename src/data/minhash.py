# minhash deduplication

import hashlib
import struct
from typing import Iterator

import numpy as np

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _sha1_hash(data: bytes) -> int:
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def _generate_hash_params(n_hashes: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _MERSENNE_PRIME, size=n_hashes, dtype=np.int64)
    b = rng.randint(0, _MERSENNE_PRIME, size=n_hashes, dtype=np.int64)
    return a, b


def shingle(text: str, k: int = 5) -> set[int]:
    """Convert text to hashed character k-shingles."""
    text = text.lower().strip()
    if len(text) < k:
        return set()
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(_sha1_hash(text[i:i + k].encode("utf-8")))
    return shingles


def minhash_signature(shingles: set[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute MinHash signature for a shingle set."""
    n_hashes = len(a)
    if not shingles:
        return np.full(n_hashes, _MAX_HASH, dtype=np.uint32)

    shingle_array = np.array(list(shingles), dtype=np.int64)
    hashes = np.mod(
        np.mod(
            a[:, np.newaxis] * shingle_array[np.newaxis, :] + b[:, np.newaxis],
            _MERSENNE_PRIME
        ),
        _MAX_HASH
    ).astype(np.uint32)
    return hashes.min(axis=1)


def jaccard_from_signatures(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


class MinHashLSH:
    """MinHash LSH index for near-duplicate detection."""

    def __init__(self, n_hashes: int = 128, n_bands: int = 16,
                 threshold: float = 0.8, shingle_k: int = 5):
        assert n_hashes % n_bands == 0
        self.n_hashes = n_hashes
        self.n_bands = n_bands
        self.rows_per_band = n_hashes // n_bands
        self.threshold = threshold
        self.shingle_k = shingle_k
        self.a, self.b = _generate_hash_params(n_hashes)
        self.buckets: list[dict[int, list[int]]] = [{} for _ in range(n_bands)]
        self.signatures: dict[int, np.ndarray] = {}
        self.n_docs = 0

    def _band_hashes(self, signature: np.ndarray) -> list[int]:
        band_hashes = []
        for i in range(self.n_bands):
            start = i * self.rows_per_band
            band = signature[start:start + self.rows_per_band]
            h = hashlib.md5(band.tobytes()).digest()
            band_hashes.append(struct.unpack("<Q", h[:8])[0])
        return band_hashes

    def insert(self, doc_id: int, text: str) -> bool:
        """Insert doc. Returns True if novel, False if near-duplicate."""
        shingles = shingle(text, self.shingle_k)
        if not shingles:
            return True

        sig = minhash_signature(shingles, self.a, self.b)
        candidates = set()
        band_hashes = self._band_hashes(sig)

        for i, bh in enumerate(band_hashes):
            candidates.update(self.buckets[i].get(bh, []))

        for cand_id in candidates:
            sim = jaccard_from_signatures(sig, self.signatures[cand_id])
            if sim >= self.threshold:
                return False

        self.signatures[doc_id] = sig
        for i, bh in enumerate(band_hashes):
            if bh not in self.buckets[i]:
                self.buckets[i][bh] = []
            self.buckets[i][bh].append(doc_id)

        self.n_docs += 1
        return True

    def stats(self) -> dict:
        total_buckets = sum(len(b) for b in self.buckets)
        return {
            "n_docs": self.n_docs,
            "n_bands": self.n_bands,
            "n_hashes": self.n_hashes,
            "rows_per_band": self.rows_per_band,
            "total_buckets": total_buckets,
            "threshold": self.threshold,
            "memory_mb": (self.n_docs * self.n_hashes * 4) / 1e6,
        }

    def clear(self):
        self.buckets = [{} for _ in range(self.n_bands)]
        self.signatures.clear()
        self.n_docs = 0


def dedup_corpus(
    documents: Iterator[tuple[int, str]],
    n_hashes: int = 128,
    n_bands: int = 16,
    threshold: float = 0.8,
    shingle_k: int = 5,
    log_interval: int = 100_000,
) -> Iterator[tuple[int, str]]:
    """Stream-dedup a corpus using MinHash LSH."""
    lsh = MinHashLSH(n_hashes=n_hashes, n_bands=n_bands,
                     threshold=threshold, shingle_k=shingle_k)
    total = 0
    kept = 0

    for doc_id, text in documents:
        total += 1
        if lsh.insert(doc_id, text):
            kept += 1
            yield doc_id, text

        if total % log_interval == 0:
            drop_pct = (total - kept) / total * 100
            stats = lsh.stats()
            print(f"  MinHash: {total:,} processed, {kept:,} kept, "
                  f"{total - kept:,} dropped ({drop_pct:.1f}%), "
                  f"index: {stats['memory_mb']:.0f} MB")

    if total > 0:
        drop_pct = (total - kept) / total * 100
        print(f"  MinHash final: {kept:,}/{total:,} kept ({drop_pct:.1f}% dropped)")
