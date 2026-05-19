"""Token-aware chunking for the SD-A-01-RAG corpus (Phase B.2).

See MASTER_PLAN.md §8 B.2.

Splitter:
    LangChain's ``RecursiveCharacterTextSplitter.from_tiktoken_encoder``.
    Recursive separator hierarchy preserves paragraph/sentence/word boundaries
    where possible. Length is measured in *tokens* (not characters), so
    ``chunk_size`` and ``chunk_overlap`` honour the §8 spec directly.

    RCTS may occasionally emit a chunk slightly longer than ``chunk_size`` if
    a single contiguous segment between separators exceeds the limit. The
    ``chunk_summary`` helper exposes per-doc max token count so this is
    visible.

Tokenizer:
    tiktoken ``cl100k_base`` as a proxy for the Llama 3.1 tokenizer. Llama
    ships its own SentencePiece tokenizer, but using it here would require
    pulling ``transformers`` and gated HF auth. For chunking we only need a
    deterministic, consistent token count — cl100k_base is within ~10% of
    Llama for English prose and is what most public RAG benchmarks use. The
    actual LLM call uses Ollama's tokenizer naturally, so this proxy only
    affects how chunks are sized, not generation correctness.

Public API:
    - :class:`Chunk` — dataclass with ``chunk_id``, ``source_doc``,
      ``start_char``, ``end_char``, ``text``, ``token_count``.
    - :func:`make_splitter` — build a token-aware RCTS.
    - :func:`chunk_document` — split one document into Chunk objects.
    - :func:`chunk_corpus` — chunk every .txt under a directory.
    - :func:`config_hash` — 8-char SHA1 of (chunk_size, overlap_pct, model).
    - :func:`save_chunks` / :func:`load_chunks` — parquet round-trip.
    - :func:`chunk_summary` — per-doc DataFrame summary.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.metrics import time_it

# --------------------------------------------------------------------------- #
# Defaults (Phase B baseline configuration — §8)                              #
# --------------------------------------------------------------------------- #

DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_OVERLAP_PCT: int = 10
DEFAULT_TOKENIZER: str = "cl100k_base"
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# Chunk dataclass                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chunk:
    """A single retrievable chunk of source text.

    Attributes:
        chunk_id: Sortable, human-readable ID, e.g. ``"01_BLOOM_0007"``.
            Format is ``"<source_doc>_<idx:04d>"``.
        source_doc: Short doc name, e.g. ``"01_BLOOM"``. Matches the file
            stem of ``corpus_text/<source_doc>.txt``.
        start_char: Inclusive char offset into the source ``.txt`` file.
        end_char: Exclusive char offset.
        text: The chunk text exactly as RCTS returned it.
        token_count: Number of tokens via tiktoken cl100k_base.
    """

    chunk_id: str
    source_doc: str
    start_char: int
    end_char: int
    text: str
    token_count: int


# --------------------------------------------------------------------------- #
# Token counting                                                              #
# --------------------------------------------------------------------------- #


def count_tokens(text: str, encoding_name: str = DEFAULT_TOKENIZER) -> int:
    """Return the token count of ``text`` under the named tiktoken encoding."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


# --------------------------------------------------------------------------- #
# Splitter construction                                                       #
# --------------------------------------------------------------------------- #


def make_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_pct: int = DEFAULT_OVERLAP_PCT,
    encoding_name: str = DEFAULT_TOKENIZER,
) -> RecursiveCharacterTextSplitter:
    """Build a token-aware RCTS at the given (chunk_size, overlap_pct) config.

    Args:
        chunk_size: Target chunk size in tokens.
        overlap_pct: Overlap as a percentage of chunk_size, e.g. ``10`` for
            10%. The overlap in tokens is ``round(chunk_size * overlap_pct /
            100)``.
        encoding_name: tiktoken encoding for length measurement.

    Returns:
        A splitter whose ``split_text`` returns a list of strings sized in
        tokens (not characters).
    """
    overlap_tokens = round(chunk_size * overlap_pct / 100)
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=overlap_tokens,
    )


# --------------------------------------------------------------------------- #
# Per-document chunking with char-offset recovery                             #
# --------------------------------------------------------------------------- #


def chunk_document(
    text: str,
    source_doc: str,
    splitter: RecursiveCharacterTextSplitter,
    encoding_name: str = DEFAULT_TOKENIZER,
) -> list[Chunk]:
    """Split a single document into Chunk objects with char offsets.

    RCTS preserves source text exactly within each chunk, so we recover
    ``start_char`` / ``end_char`` by linear scan of the source string. The
    scan advances monotonically (each chunk's start is searched from
    ``previous_start + 1``), which matches RCTS's left-to-right output order
    and is robust to chunk overlap.

    Args:
        text: The full document text.
        source_doc: Short doc name written into ``chunk_id`` and
            ``source_doc`` fields.
        splitter: Configured RCTS (from :func:`make_splitter`).
        encoding_name: tiktoken encoding for ``token_count`` measurement.

    Returns:
        List of Chunk objects, ordered by appearance in source.

    Raises:
        RuntimeError: If a chunk cannot be located in the source text.
            This signals that the splitter rewrote whitespace — fail loudly
            rather than silently misalign offsets that Phase E recall@k
            depends on.
    """
    enc = tiktoken.get_encoding(encoding_name)
    pieces = splitter.split_text(text)

    chunks: list[Chunk] = []
    search_from = 0
    for i, piece in enumerate(pieces):
        pos = text.find(piece, search_from)
        if pos == -1:
            preview = piece[:80].replace("\n", "\\n")
            raise RuntimeError(
                f"chunk_document: could not locate piece {i} of "
                f"'{source_doc}' in source text. Preview: {preview!r}..."
            )
        chunks.append(
            Chunk(
                chunk_id=f"{source_doc}_{i:04d}",
                source_doc=source_doc,
                start_char=pos,
                end_char=pos + len(piece),
                text=piece,
                token_count=len(enc.encode(piece)),
            )
        )
        # Advance by 1 so the next find skips the just-found chunk's start
        # but can still match an overlapping chunk that begins inside this
        # chunk's body. Exact-duplicate chunks (rare for 512-token windows
        # on prose) would alias here; we rely on RCTS's left-to-right output
        # order to keep alignment correct.
        search_from = pos + 1
    return chunks


# --------------------------------------------------------------------------- #
# Corpus-level chunking                                                       #
# --------------------------------------------------------------------------- #


@time_it("chunking")
def chunk_corpus(
    text_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_pct: int = DEFAULT_OVERLAP_PCT,
    encoding_name: str = DEFAULT_TOKENIZER,
    sources: Iterable[str] | None = None,
) -> list[Chunk]:
    """Chunk every ``.txt`` under ``text_dir`` and return a flat list.

    Args:
        text_dir: Directory containing ``<source_doc>.txt`` files.
        chunk_size: See :func:`make_splitter`.
        overlap_pct: See :func:`make_splitter`.
        encoding_name: tiktoken encoding for chunking *and* token counting.
        sources: Optional iterable of short doc names to restrict to (e.g.
            ``["01_BLOOM"]``). Defaults to every ``.txt`` file in
            ``text_dir``, sorted lexicographically.

    Returns:
        Flat list of Chunk objects across all docs, in source-doc order.
    """
    text_dir = Path(text_dir)
    if sources is None:
        files = sorted(text_dir.glob("*.txt"))
    else:
        files = [text_dir / f"{name}.txt" for name in sources]

    splitter = make_splitter(chunk_size, overlap_pct, encoding_name)
    all_chunks: list[Chunk] = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"chunk_corpus: missing text file {f}")
        source_doc = f.stem
        text = f.read_text(encoding="utf-8")
        all_chunks.extend(
            chunk_document(text, source_doc, splitter, encoding_name)
        )
    return all_chunks


# --------------------------------------------------------------------------- #
# Configuration hashing (§15)                                                 #
# --------------------------------------------------------------------------- #


def config_hash(
    chunk_size: int,
    overlap_pct: int,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> str:
    """8-char SHA1 hash of the (chunk_size, overlap_pct, embedding_model) tuple.

    Used as a folder name under ``indices/``. Stable across runs and OSes.
    See MASTER_PLAN.md §15.

    Args:
        chunk_size: Chunk size in tokens.
        overlap_pct: Overlap as a percentage of chunk_size.
        embedding_model: Full embedding-model identifier (e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``).

    Returns:
        First 8 characters of the SHA1 hex digest.
    """
    key = (
        f"chunk_size={chunk_size}|overlap_pct={overlap_pct}|"
        f"embedding={embedding_model}"
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def save_chunks(chunks: list[Chunk], parquet_path: Path) -> None:
    """Persist a list of Chunk objects to parquet.

    The parent directory is created if missing.
    """
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(c) for c in chunks])
    df.to_parquet(parquet_path, index=False)


def load_chunks(parquet_path: Path) -> list[Chunk]:
    """Load chunks from parquet back into Chunk dataclasses.

    Numeric columns are cast to native ``int`` so equality comparisons with
    ints elsewhere in the pipeline behave consistently.
    """
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    chunks: list[Chunk] = []
    for row in df.to_dict(orient="records"):
        chunks.append(
            Chunk(
                chunk_id=str(row["chunk_id"]),
                source_doc=str(row["source_doc"]),
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                text=str(row["text"]),
                token_count=int(row["token_count"]),
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #


def chunk_summary(chunks: list[Chunk]) -> pd.DataFrame:
    """Per-doc summary DataFrame.

    Columns:
        source_doc, n_chunks, mean_tokens, min_tokens, max_tokens,
        p95_tokens, mean_chars.

    The last row aggregates across all docs and has ``source_doc == "TOTAL"``.
    """
    df = pd.DataFrame([asdict(c) for c in chunks])
    by_doc = (
        df.groupby("source_doc")
        .agg(
            n_chunks=("chunk_id", "count"),
            mean_tokens=("token_count", "mean"),
            min_tokens=("token_count", "min"),
            max_tokens=("token_count", "max"),
            p95_tokens=("token_count", lambda s: s.quantile(0.95)),
            mean_chars=("text", lambda s: s.str.len().mean()),
        )
        .reset_index()
    )
    total_row = pd.DataFrame(
        [
            {
                "source_doc": "TOTAL",
                "n_chunks": int(by_doc["n_chunks"].sum()),
                "mean_tokens": float(df["token_count"].mean()),
                "min_tokens": int(df["token_count"].min()),
                "max_tokens": int(df["token_count"].max()),
                "p95_tokens": float(df["token_count"].quantile(0.95)),
                "mean_chars": float(df["text"].str.len().mean()),
            }
        ]
    )
    return pd.concat([by_doc, total_row], ignore_index=True)


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP_PCT",
    "DEFAULT_TOKENIZER",
    "DEFAULT_EMBEDDING_MODEL",
    "Chunk",
    "count_tokens",
    "make_splitter",
    "chunk_document",
    "chunk_corpus",
    "config_hash",
    "save_chunks",
    "load_chunks",
    "chunk_summary",
]
