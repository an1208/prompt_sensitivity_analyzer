"""Text embedders. All return L2-normalised float arrays, so a dot product is cosine similarity."""
from __future__ import annotations

import math
import re
import zlib
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(ABC):
    name: str = "embedder"

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:  # (n, d), rows unit-norm (or zero)
        ...


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = DEFAULT_ST_MODEL, device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers` "
                "or use `--embedder hash` for the offline lexical fallback."
            ) from e
        self.name = f"sentence-transformers:{model_name}"
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        arr = self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(arr, dtype=np.float64)


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free bag-of-(uni+bi)grams with signed feature hashing.

    Captures lexical overlap only (no semantics) - use it for tests, offline runs and as a
    sanity-check baseline against the neural embedder.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for r, text in enumerate(texts):
            toks = re.findall(r"[a-z0-9']+", text.lower())
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for feat, cnt in Counter(feats).items():
                b = feat.encode()
                idx = zlib.crc32(b) % self.dim
                sign = 1.0 if (zlib.crc32(b + b"#") & 1) else -1.0
                out[r, idx] += sign * (1.0 + math.log(cnt))
            n = np.linalg.norm(out[r])
            if n > 0:
                out[r] /= n
        return out


def make_embedder(kind: str = "st", model: Optional[str] = None) -> Embedder:
    if kind == "hash":
        return HashingEmbedder()
    if kind == "st":
        return SentenceTransformerEmbedder(model or DEFAULT_ST_MODEL)
    raise ValueError(f"unknown embedder {kind!r} (choose 'st' or 'hash')")
