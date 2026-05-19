"""FAISS indexing for the SD-A-01-RAG corpus (Phase B.4).

See MASTER_PLAN.md §8 B.4.

Index choice: ``IndexFlatIP`` over L2-normalized embeddings. With normalized
vectors, inner product equals cosine similarity, so this is a standard
exhaustive cosine-similarity search. Exact (no approximation) — appropriate
for our corpus size of ~500-700 chunks. ANN structures like IVF or HNSW would
add a quality loss without meaningful speedup at this scale.

Persistence: an index folder under ``indices/<config_hash>/`` contains:

    - ``faiss.index``    — binary FAISS index file
    - ``chunks.parquet`` — the chunks that align row-for-row with the index

Chunks and the index must always be loaded together; their order is the
source of truth for which row corresponds to which chunk.

Public API:
    - :func:`build_index` — build an IndexFlatIP from float32 embeddings.
    - :func:`save_index` — persist (index, chunks) to a folder.
    - :func:`load_index` — load (index, chunks) from a folder.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from src.chunking import Chunk, load_chunks, save_chunks
from src.metrics import time_it


@time_it("faiss_build")
def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an ``IndexFlatIP`` from L2-normalized embeddings.

    Args:
        embeddings: ``(n, d)`` array, expected to be L2-normalized. Will be
            cast to ``float32`` if it is not already.

    Returns:
        A FAISS ``IndexFlatIP`` populated with all rows of ``embeddings``.

    Raises:
        ValueError: If ``embeddings`` is not 2-D.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"build_index: embeddings must be 2-D, got shape {embeddings.shape}"
        )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


@time_it("index_save")
def save_index(
    index: faiss.Index,
    chunks: list[Chunk],
    out_dir: Path,
) -> None:
    """Persist a (FAISS index, chunks) pair to ``out_dir``.

    Writes:
        - ``out_dir / "faiss.index"``
        - ``out_dir / "chunks.parquet"``

    The output directory is created if missing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    save_chunks(chunks, out_dir / "chunks.parquet")


@time_it("index_load")
def load_index(out_dir: Path) -> tuple[faiss.Index, list[Chunk]]:
    """Load a (FAISS index, chunks) pair previously saved by :func:`save_index`.

    Raises:
        RuntimeError: If the number of vectors in the index does not match
            the number of chunks in the parquet — that would silently
            corrupt retrieval results.
    """
    out_dir = Path(out_dir)
    index = faiss.read_index(str(out_dir / "faiss.index"))
    chunks = load_chunks(out_dir / "chunks.parquet")
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"load_index: vector/chunk count mismatch at {out_dir}: "
            f"{index.ntotal} vectors vs. {len(chunks)} chunks."
        )
    return index, chunks


__all__ = [
    "build_index",
    "save_index",
    "load_index",
]
