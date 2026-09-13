"""Embedding generation (offline-first).

The DEFAULT embedder is a deterministic bag-of-words hasher built on NumPy: each
token is hashed into one of ``dim`` buckets, counts are accumulated, and the
vector is L2-normalised. It needs no model download and no network, so retrieval
is fully reproducible and testable offline. Two texts sharing many words end up
with vectors pointing in a similar direction, so cosine similarity ranks
word overlap, which is enough to make retrieval meaningful.

Stopwords are dropped before hashing. This matters more than it looks: in a
bag-of-words vector the function words ("what", "is", "the", "of") are the most
frequent tokens in BOTH the question and every chunk, so leaving them in makes
every chunk look similar to every question. Measured on the sample corpus, the
question "what is the capital of france" scored 0.228 cosine against a leave
policy chunk while the on-topic question scored 0.254 -- almost no separation.
Removing stopwords makes the cosine reflect content-word overlap only, which is
what downstream confidence scoring relies on.

A real ``sentence-transformers`` model can be dropped in later behind the same
interface; it is intentionally not required for the default path or the tests.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from ragfabric_core.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately small, hand-written English stopword list. A big curated list
# would be a dependency; these are the function words that actually distort a
# bag-of-words vector because they appear in nearly every sentence.
STOPWORDS = frozenset(
    """
    a about an and any are as at be been being but by can could did do does
    doing done for from had has have having he her hers him his how i if in
    into is it its me my no nor not of on or our ours out over own she should
    so some such than that the their theirs them then there these they this
    those to too us was we were what when where which while who whom why will
    with would you your yours
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric word tokenizer (raw, stopwords included)."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed, used for embedding and lexical scoring.

    Falls back to the raw tokens when a string is *entirely* stopwords (e.g. the
    query "what is it"), because an all-zero vector would score 0 against
    everything and give the caller no ranking at all.
    """
    tokens = tokenize(text)
    kept = [t for t in tokens if t not in STOPWORDS]
    return kept or tokens


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for empty text
    return matrix / norms


class HashingEmbedder:
    """Deterministic bag-of-words embedder via the hashing trick (DEFAULT)."""

    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.embedding_dim
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    def _hash(self, token: str) -> int:
        # Use a stable hash (md5) rather than the salted built-in hash() so
        # embeddings are reproducible across processes and runs.
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % self.dim

    def embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in content_tokens(text):
            vector[self._hash(token)] += 1.0
        return _l2_normalize(vector.reshape(1, -1))[0]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed_one(t) for t in texts]).astype(np.float32)


# Process-wide default embedder (cheap to construct, but shared for clarity).
_default_embedder: HashingEmbedder | None = None


def get_embedder() -> HashingEmbedder:
    """Return the shared default (offline) embedder."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = HashingEmbedder()
    return _default_embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts and return plain lists of floats (JSON-serialisable)."""
    return [vec.tolist() for vec in get_embedder().embed(texts)]
