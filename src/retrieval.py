"""Top-k retrieval for the SD-A-01-RAG corpus (Phase B.5).

See MASTER_PLAN.md §8 B.5.

The retrieve() function is a one-shot query → top-k API. It embeds the query
internally so callers do not need to know about the embedding layer. For
finer-grained timing (separating the embed from the index search), see
``src/pipeline.py`` which inlines the two steps.

Public API:
    - :class:`RetrievalResult` — a chunk paired with its similarity score.
    - :func:`retrieve` — one-shot top-k retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from src.chunking import Chunk, DEFAULT_EMBEDDING_MODEL
from src.embedding import embed_query
from src.metrics import time_it


DEFAULT_TOP_K: int = 3


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieved chunk paired with its similarity score and rank.

    Attributes:
        chunk: The retrieved Chunk.
        score: Cosine similarity (because vectors are L2-normalized and the
            FAISS index is IndexFlatIP). Higher is more similar.
        rank: 0-indexed rank within the top-k result set.
    """

    chunk: Chunk
    score: float
    rank: int


@time_it("retrieve_oneshot")
def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[Chunk],
    k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[RetrievalResult]:
    """Embed ``query``, run top-k cosine search, return ranked results.

    Args:
        query: Natural-language query.
        index: FAISS ``IndexFlatIP`` from :func:`src.indexing.build_index`.
        chunks: The chunks list aligned row-for-row with ``index``.
        k: How many top chunks to return. If ``k > index.ntotal`` FAISS pads
            with ``-1`` indices, which are filtered out.
        model_name: Embedding model for the query. Must match the model used
            to embed the chunks; mismatch will silently produce nonsense
            because the vector spaces are different.

    Returns:
        List of :class:`RetrievalResult`, ordered best-first (highest score
        first). Length is at most ``k``.
    """
    q_vec = embed_query(query, model_name=model_name)
    q = q_vec.reshape(1, -1).astype(np.float32, copy=False)
    scores, idxs = index.search(q, k)

    results: list[RetrievalResult] = []
    for rank, (score, idx) in enumerate(
        zip(scores[0].tolist(), idxs[0].tolist())
    ):
        if idx < 0:
            continue  # FAISS pads with -1 if k > ntotal
        results.append(
            RetrievalResult(chunk=chunks[idx], score=float(score), rank=rank)
        )
    return results


__all__ = [
    "DEFAULT_TOP_K",
    "RetrievalResult",
    "retrieve",
]
