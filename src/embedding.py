"""Sentence-transformers wrapper for the SD-A-01-RAG corpus (Phase B.3).

See MASTER_PLAN.md §8 B.3.

Loaded models are cached at module level so repeated calls reuse the same
``SentenceTransformer`` instance — important because the first load can take
several seconds even when the model file is already cached on disk.

Embeddings are L2-normalized at encode time (via ``normalize_embeddings=True``
in ``SentenceTransformer.encode``) so a FAISS ``IndexFlatIP`` gives cosine
similarity (§8 vector-store choice). Output dtype is ``float32`` to match
FAISS's expected input.

Public API:
    - :func:`get_model` — load (or fetch from cache) a SentenceTransformer.
    - :func:`embed_texts` — encode a list of strings.
    - :func:`embed_chunks` — encode a list of Chunk objects (uses ``.text``).
    - :func:`embed_query` — encode a single query string.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk, DEFAULT_EMBEDDING_MODEL
from src.metrics import time_it

# --------------------------------------------------------------------------- #
# Module-level model cache                                                    #
# --------------------------------------------------------------------------- #

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load (or fetch from cache) a SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier, e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``.

    Returns:
        A ready-to-use ``SentenceTransformer`` instance, kept alive by the
        module-level cache.
    """
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


# --------------------------------------------------------------------------- #
# Encoding                                                                    #
# --------------------------------------------------------------------------- #


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode a list of strings to L2-normalized float32 embeddings.

    Args:
        texts: List of strings to embed.
        model_name: Embedding-model identifier.
        batch_size: SentenceTransformer batch size.
        show_progress: Whether to show the tqdm progress bar.

    Returns:
        ``(n, d)`` float32 array of L2-normalized embeddings.
    """
    model = get_model(model_name)
    arr = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return arr.astype(np.float32, copy=False)


@time_it("embedding")
def embed_chunks(
    chunks: list[Chunk],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed a list of Chunk objects (using ``Chunk.text``).

    Returns:
        ``(n_chunks, dim)`` float32 array, L2-normalized.
    """
    return embed_texts(
        [c.text for c in chunks],
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
    )


def embed_query(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """Embed a single query string.

    Returns:
        ``(dim,)`` float32 vector, L2-normalized.
    """
    arr = embed_texts(
        [text],
        model_name=model_name,
        batch_size=1,
        show_progress=False,
    )
    return arr[0]


__all__ = [
    "get_model",
    "embed_texts",
    "embed_chunks",
    "embed_query",
]
