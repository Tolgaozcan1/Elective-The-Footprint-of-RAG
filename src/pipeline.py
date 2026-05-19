"""End-to-end RAG pipeline composer (Phase B.7).

See MASTER_PLAN.md §8 B.7.

This module ties B.3 (embedding) + B.5 (retrieval) + B.6 (generation) into a
single ``run_rag`` function that takes a query and returns the answer plus
diagnostics. It deliberately re-implements the embed-then-search sequence
inline (rather than calling ``retrieve``) so the embedding step and the FAISS
search can be timed separately — that distinction is needed for Phase D's
cost analysis.

**Phase B vs. Phase C:** the timings dict here uses ``time.perf_counter`` and
captures wall-clock seconds only. Phase C will add CodeCarbon, peak RAM, and
token counts via ``src/metrics.py``.

Public API:
    - :class:`RAGConfig` — one-shot configuration dataclass.
    - :func:`run_rag` — run a single query against a pre-built (index, chunks).
    - :func:`run_rag_from_dir` — convenience: load (index, chunks) from disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OVERLAP_PCT,
    Chunk,
    config_hash,
)
from src.embedding import embed_query
from src.generation import (
    DEFAULT_LLM,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    build_prompt,
    generate_from_prompt,
)
from src.indexing import load_index
from src.retrieval import DEFAULT_TOP_K, RetrievalResult


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class RAGConfig:
    """Configuration for a single end-to-end RAG run.

    Defaults match the §8 Phase B baseline configuration. Phase D experiments
    sweep ``chunk_size``, ``overlap_pct``, ``embedding_model``, and ``top_k``
    while keeping everything else fixed.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap_pct: int = DEFAULT_OVERLAP_PCT
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = DEFAULT_TOP_K
    llm_model: str = DEFAULT_LLM
    temperature: float = DEFAULT_TEMPERATURE
    seed: int = DEFAULT_SEED

    @property
    def hash(self) -> str:
        """8-char SHA1 of (chunk_size, overlap_pct, embedding_model). See §15."""
        return config_hash(
            self.chunk_size, self.overlap_pct, self.embedding_model
        )


# --------------------------------------------------------------------------- #
# End-to-end run                                                              #
# --------------------------------------------------------------------------- #


def run_rag(
    query: str,
    config: RAGConfig,
    index: faiss.Index,
    chunks: list[Chunk],
) -> dict[str, Any]:
    """Run one query end-to-end: embed → retrieve → generate.

    The (index, chunks) pair must already exist; build it once per
    configuration via :func:`src.indexing.build_index` /
    :func:`src.indexing.save_index`. For convenience, see
    :func:`run_rag_from_dir` which loads them from
    ``indices/<config_hash>/``.

    Args:
        query: Natural-language question.
        config: :class:`RAGConfig` (defaults match §8 baseline).
        index: FAISS index aligned row-for-row with ``chunks``.
        chunks: Chunks list aligned row-for-row with ``index``.

    Returns:
        Dict with keys:

        - ``query`` (str): the input query.
        - ``retrieved_chunks`` (list[RetrievalResult]): top-k results.
        - ``prompt`` (str): the fully formatted prompt sent to the LLM
          (Phase C addition — used by ``src/metrics.py`` to count prompt
          tokens without rebuilding the prompt).
        - ``answer`` (str): generated answer text (whitespace-stripped).
        - ``timings`` (dict[str, float]): wall-clock seconds for ``embed``,
          ``retrieve``, ``generate``.
    """
    timings: dict[str, float] = {}

    # B.3 — embed the query.
    t0 = time.perf_counter()
    q_vec = embed_query(query, model_name=config.embedding_model)
    timings["embed"] = time.perf_counter() - t0

    # B.5 — search the FAISS index.
    t0 = time.perf_counter()
    q = q_vec.reshape(1, -1).astype(np.float32, copy=False)
    scores, idxs = index.search(q, config.top_k)
    retrieved: list[RetrievalResult] = [
        RetrievalResult(chunk=chunks[idx], score=float(s), rank=r)
        for r, (s, idx) in enumerate(
            zip(scores[0].tolist(), idxs[0].tolist())
        )
        if idx >= 0
    ]
    timings["retrieve"] = time.perf_counter() - t0

    # B.6 — build prompt + generate the answer.
    # Phase C: build_prompt is called *exactly once* here so the prompt text
    # can be returned for token counting without re-building it. The
    # resulting prompt is byte-identical to what the pre-refactor
    # ``generate(query, retrieved, ...)`` would have produced — same
    # ``build_prompt`` function, same arguments.
    prompt = build_prompt(query, retrieved)
    t0 = time.perf_counter()
    answer = generate_from_prompt(
        prompt,
        model_name=config.llm_model,
        temperature=config.temperature,
        seed=config.seed,
    )
    timings["generate"] = time.perf_counter() - t0

    return {
        "query": query,
        "retrieved_chunks": retrieved,
        "prompt": prompt,
        "answer": answer,
        "timings": timings,
    }


def run_rag_from_dir(
    query: str,
    index_dir: Path,
    config: RAGConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: load (index, chunks) from ``index_dir``, then run.

    Args:
        query: Natural-language question.
        index_dir: Directory previously written by
            :func:`src.indexing.save_index`, e.g.
            ``indices/<config_hash>/``.
        config: :class:`RAGConfig`. Defaults to the §8 baseline.

    Returns:
        Same dict shape as :func:`run_rag`.
    """
    config = config or RAGConfig()
    index, chunks = load_index(index_dir)
    return run_rag(query, config, index, chunks)


__all__ = [
    "RAGConfig",
    "run_rag",
    "run_rag_from_dir",
]
