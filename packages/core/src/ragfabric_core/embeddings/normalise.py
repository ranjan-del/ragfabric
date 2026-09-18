"""L2 normalisation, applied once at write time.

Why here and not inside each store: pgvector ranks with cosine distance and the
SQLite fallback ranks with a dot product. Those two agree only when the vectors
are unit length. Normalising on the way in makes every store, every dialect and
the reindexer score identically, and makes a stored score directly comparable
across models.
"""

from __future__ import annotations

import math


def normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm == 0.0:
        return [float(v) for v in vector]
    return [float(v) / norm for v in vector]


def normalise_all(vectors: list[list[float]]) -> list[list[float]]:
    return [normalise(v) for v in vectors]
