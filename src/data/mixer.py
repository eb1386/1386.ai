# data source mixing

import math
from pathlib import Path
from typing import Iterator

import numpy as np


class DataSource:
    """Single data source with weight."""

    def __init__(self, name: str, path: Path, weight: float,
                 upsample_factor: float = 1.0):
        self.name = name
        self.path = path
        self.weight = weight
        self.upsample_factor = upsample_factor
        self._size_bytes = path.stat().st_size if path.exists() else 0

    @property
    def size_gb(self) -> float:
        return self._size_bytes / 1e9

    def iter_documents(self) -> Iterator[str]:
        if not self.path.exists():
            return
        buffer = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "" and buffer:
                    text = "\n".join(buffer).strip()
                    if len(text) > 50:
                        yield text
                    buffer = []
                else:
                    buffer.append(line.rstrip("\n"))
            if buffer:
                text = "\n".join(buffer).strip()
                if len(text) > 50:
                    yield text

    def __repr__(self):
        return f"DataSource({self.name!r}, {self.size_gb:.2f} GB, weight={self.weight})"


class DataMixer:
    """Mix sources according to target ratios."""

    def __init__(self, sources: list[DataSource], seed: int = 42):
        self.sources = sources
        self.seed = seed
        total_weight = sum(s.weight for s in sources)
        for s in sources:
            s.weight = s.weight / total_weight

    def mix(self, target_chars: int | None = None) -> Iterator[tuple[str, str]]:
        """Yield (source_name, text) in mixed order."""
        rng = np.random.default_rng(self.seed)

        source_sizes = [s._size_bytes for s in self.sources]
        total_available = sum(source_sizes)
        if target_chars is None:
            target_chars = total_available

        for s, size in zip(self.sources, source_sizes):
            target = int(target_chars * s.weight)
            if size > 0 and target > size:
                s.upsample_factor = math.ceil(target / size)

        source_iters = {s.name: self._iter_with_upsample(s) for s in self.sources}
        source_chars = {s.name: 0 for s in self.sources}
        exhausted = set()
        total_yielded = 0

        while total_yielded < target_chars and len(exhausted) < len(self.sources):
            best = None
            best_deficit = -float("inf")
            for s in self.sources:
                if s.name in exhausted:
                    continue
                actual = source_chars[s.name] / max(1, total_yielded)
                deficit = s.weight - actual
                if deficit > best_deficit:
                    best_deficit = deficit
                    best = s

            if best is None:
                break

            try:
                doc = next(source_iters[best.name])
                yield best.name, doc
                n = len(doc)
                source_chars[best.name] += n
                total_yielded += n
            except StopIteration:
                exhausted.add(best.name)

    def _iter_with_upsample(self, source: DataSource) -> Iterator[str]:
        for _ in range(max(1, int(source.upsample_factor))):
            yield from source.iter_documents()

    def summary(self) -> str:
        lines = ["Data mix:"]
        for s in self.sources:
            up = f"  [upsample {int(s.upsample_factor)}x]" if s.upsample_factor > 1 else ""
            lines.append(f"  {s.name:20s}  {s.weight:5.1%}  ({s.size_gb:.2f} GB){up}")
        return "\n".join(lines)
