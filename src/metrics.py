"""Measurement primitives for the SD-A-01-RAG project (Phase C).

See MASTER_PLAN.md §9 (Phase C — Measurement Layer) and §13.3 (indexing vs.
query-time cost separation).

This module provides four orthogonal building blocks that together populate
``results/indexing_log.csv`` and ``results/query_log.csv``:

C.1 — Wall-clock timing
    :func:`time_it` is a *zero-overhead-when-inactive* decorator. Outside a
    :func:`timing_context` it is a passthrough; inside one, it accumulates
    wall-clock seconds into a thread-local dict keyed by label. This means
    decorating ``src/`` functions does not change Phase B behaviour — the
    timing only "turns on" when a notebook explicitly opens a context.

    Per-chain decorating rule (avoid double-counting along a call chain):
    decorate exactly one entry per call chain. See PROGRESS_LOG.md Phase C
    entry for the chosen labels.

C.2 — Peak RAM
    :class:`PeakRAMSampler` polls ``psutil.Process().memory_info().rss`` from a
    daemon thread at 100 ms cadence and records the maximum observed RSS. It
    captures the entire process (Python heap + PyTorch C++ allocations + FAISS
    C++ allocations + shared libs), which is what the §9 ``peak_ram_mb``
    column expects.

C.3 — Token counting
    Helpers using tiktoken ``cl100k_base`` — the *same* encoder used by Phase B
    chunking (``src/chunking.py``). Matching the chunking encoder is required
    by §9 C.3 ("match what was used in Phase B chunking for consistency").

C.5 — Determinism guards
    :func:`set_global_seeds` seeds ``random``, ``numpy``, and ``torch``
    (if importable). Notebook-side equivalent of the ``temperature=0,
    seed=42`` already enforced as Ollama defaults in ``src/generation.py``.

Public API:
    - :func:`timing_context` — context manager, yields a dict[str, float].
    - :func:`time_it` — decorator factory.
    - :class:`PeakRAMSampler` — peak-RSS sampler context manager.
    - :func:`count_retrieved_tokens` — sum of pre-computed chunk token counts.
    - :func:`count_text_tokens` — token count of an arbitrary string.
    - :func:`set_global_seeds` — seed random / numpy / torch.
"""

from __future__ import annotations

import functools
import random
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, TypeVar

import numpy as np
import psutil

# NOTE: ``src.chunking`` is imported lazily inside :func:`count_text_tokens`
# to break the circular import — ``src/chunking.py`` (and other ``src/``
# modules) import :func:`time_it` from this module.

if TYPE_CHECKING:
    from src.retrieval import RetrievalResult

# --------------------------------------------------------------------------- #
# C.1 — Timing                                                                #
# --------------------------------------------------------------------------- #

# Thread-local stack of timing dicts. A *stack* (not a single dict) so nested
# ``with timing_context()`` blocks remain isolated — handy for tests, and a
# defensive choice if Phase D nests indexing inside a sweep loop.
_state = threading.local()

F = TypeVar("F", bound=Callable[..., Any])


def _stack() -> list[dict[str, float]]:
    """Lazy-init the per-thread timing stack."""
    if not hasattr(_state, "stack"):
        _state.stack = []
    return _state.stack  # type: ignore[no-any-return]


@contextmanager
def timing_context() -> Iterator[dict[str, float]]:
    """Push a fresh timing dict onto the thread-local stack.

    Inside the ``with`` block, every call to a ``@time_it``-decorated function
    accumulates wall-clock seconds into the yielded dict, keyed by the
    decorator's label.

    Example::

        with timing_context() as timings:
            chunks = chunk_corpus(...)
            embs = embed_chunks(chunks)
        print(timings)  # {"chunking": 1.23, "embedding": 28.4}

    Yields:
        The timing dict that ``time_it`` writes into. Mutated in place; safe
        to read after the block exits.
    """
    timings: dict[str, float] = {}
    _stack().append(timings)
    try:
        yield timings
    finally:
        _stack().pop()


def time_it(label: str | None = None) -> Callable[[F], F]:
    """Wall-clock timing decorator. Zero overhead when no context is active.

    The decorated function:

    - Outside any :func:`timing_context`: behaves exactly like the undecorated
      function — one ``hasattr`` lookup and one ``if`` test, no clock reads.
      This keeps Phase B's observed performance intact.
    - Inside a :func:`timing_context`: brackets the call with
      ``time.perf_counter`` and adds the elapsed seconds to the active
      thread-local dict under ``label`` (or ``fn.__qualname__`` if label is
      ``None``). Cumulative across calls within a context.

    Args:
        label: Key used in the timing dict. Defaults to ``fn.__qualname__``.

    Returns:
        A decorator. Apply as ``@time_it("embedding")``.
    """
    def decorator(fn: F) -> F:
        key = label if label is not None else fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            stack = _stack()
            if not stack:  # zero-overhead Phase B path
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                stack[-1][key] = stack[-1].get(key, 0.0) + (
                    time.perf_counter() - t0
                )

        return wrapper  # type: ignore[return-value]

    return decorator


# --------------------------------------------------------------------------- #
# C.2 — Peak RAM                                                              #
# --------------------------------------------------------------------------- #


class PeakRAMSampler:
    """Background-sampled peak resident-set size, in megabytes (MB).

    Used as a context manager. The sampler thread is a daemon so it never
    blocks interpreter shutdown. Sampling cadence defaults to 100 ms — enough
    to catch peak transients during ~30 s indexing runs without measurable
    overhead.

    The reported ``peak_rss_mb`` is the maximum *whole-process* RSS observed
    between ``__enter__`` and ``__exit__`` (inclusive of one final sample at
    exit). This captures Python heap + PyTorch ATen tensor allocations +
    FAISS C++ allocations + shared libraries — i.e. the number that matters
    for §9 ``peak_ram_mb``.

    Note on units: 1 MB = 1_000_000 bytes (decimal MB), to match what
    ``codecarbon`` and most Linux tooling use. ``rss`` from psutil is in
    bytes.

    Example::

        with PeakRAMSampler() as ram:
            embs = embed_chunks(chunks)
        print(ram.peak_rss_mb, ram.delta_rss_mb)
    """

    def __init__(self, interval_sec: float = 0.1) -> None:
        if interval_sec <= 0:
            raise ValueError(
                f"PeakRAMSampler.interval_sec must be > 0, got {interval_sec}"
            )
        self.interval_sec = interval_sec
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_rss_mb: float = 0.0
        self.peak_rss_mb: float = 0.0

    # -- context-manager protocol -------------------------------------------

    def __enter__(self) -> "PeakRAMSampler":
        self.baseline_rss_mb = self._rss_mb()
        self.peak_rss_mb = self.baseline_rss_mb
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # One final sample at exit so we never miss a tail spike.
        rss = self._rss_mb()
        if rss > self.peak_rss_mb:
            self.peak_rss_mb = rss

    # -- internals -----------------------------------------------------------

    def _rss_mb(self) -> float:
        return self._proc.memory_info().rss / 1_000_000.0

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            rss = self._rss_mb()
            if rss > self.peak_rss_mb:
                self.peak_rss_mb = rss
            # Event.wait returns True if signalled; either way we loop check.
            self._stop.wait(self.interval_sec)

    # -- read-only derived values -------------------------------------------

    @property
    def delta_rss_mb(self) -> float:
        """Peak − baseline RSS, in MB."""
        return self.peak_rss_mb - self.baseline_rss_mb


# --------------------------------------------------------------------------- #
# C.3 — Token counting                                                        #
# --------------------------------------------------------------------------- #


def count_text_tokens(
    text: str,
    encoding_name: str | None = None,
) -> int:
    """Token count of an arbitrary string under tiktoken ``cl100k_base``.

    Thin wrapper over :func:`src.chunking.count_tokens` so all token-counting
    in the measurement layer routes through one swappable function — useful
    if Phase D ever wants to compare encoders.

    Args:
        text: Any string (prompt, generated answer, etc.).
        encoding_name: tiktoken encoding name. Defaults to the chunking
            encoder (``DEFAULT_TOKENIZER``) for consistency with Phase B
            (§9 C.3).

    Returns:
        Number of tokens.
    """
    # Lazy import — see module-level note about the circular dependency.
    from src.chunking import DEFAULT_TOKENIZER, count_tokens

    return count_tokens(
        text, encoding_name=encoding_name or DEFAULT_TOKENIZER
    )


def count_retrieved_tokens(retrieved: Sequence["RetrievalResult"]) -> int:
    """Sum of pre-computed token counts on the retrieved chunks.

    Each ``Chunk`` already carries a ``token_count`` field set at chunking
    time (under the same ``cl100k_base`` encoder used elsewhere in the
    measurement layer), so this is O(k) and never re-tokenises.

    Args:
        retrieved: Iterable of :class:`src.retrieval.RetrievalResult`.

    Returns:
        Sum of ``r.chunk.token_count`` over all retrieved results.
    """
    return sum(int(r.chunk.token_count) for r in retrieved)


# --------------------------------------------------------------------------- #
# C.5 — Determinism guards                                                    #
# --------------------------------------------------------------------------- #


def set_global_seeds(seed: int = 42) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (if importable).

    Call this once at the top of every notebook (§9 C.5). The Ollama-side
    determinism (``temperature=0``, ``seed=42``) is enforced separately via
    :data:`src.generation.DEFAULT_TEMPERATURE` and
    :data:`src.generation.DEFAULT_SEED`, which every Ollama call already
    honours.

    Args:
        seed: Seed value. Default 42 (project-wide convention, §15).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is a transitive dependency of sentence-transformers
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# C.6 — Result-log helpers (extracted from notebooks/02_measurement_layer.ipynb)
# --------------------------------------------------------------------------- #
#
# Phase C cell 10 ("schema lock-in + sanity asserts A–H") and cell 8
# ("proportional energy allocation") are lifted here so every Phase D notebook
# (`03_exp1_chunk_overlap.ipynb`, `04_exp2_embedding_model.ipynb`,
# `05_exp3_topk_depth.ipynb`) can re-use them without copy-paste. This is the
# explicit "Next" item of the Phase C entry in PROGRESS_LOG.md.
#
# The §9 schemas are encoded as module-level constants. Any change to the §9
# log schema is a one-line edit here that all callers automatically pick up.

# §9 schema — indexing_log.csv
EXPECTED_INDEXING_COLS: list[str] = [
    "run_id", "timestamp", "config_hash", "chunk_size", "chunk_overlap",
    "embedding_model", "n_documents", "n_chunks_total", "indexing_time_sec",
    "peak_ram_mb", "index_size_mb", "embedding_time_sec",
    "faiss_build_time_sec", "energy_wh", "co2_g", "notes",
]

# §9 schema — query_log.csv
EXPECTED_QUERY_COLS: list[str] = [
    "run_id", "timestamp", "config_hash", "question_id", "top_k",
    "query_embed_time_ms", "retrieval_time_ms", "generation_time_ms",
    "total_latency_ms", "n_retrieved_chunks", "retrieved_token_count",
    "prompt_token_count", "answer_token_count", "energy_wh_per_query",
    "co2_g_per_query", "retrieved_chunk_ids", "answer_text", "notes",
]

# Phase C Q2 tag — every row whose energy_wh_per_query / co2_g_per_query was
# computed by :func:`allocate_block_energy_proportionally` must carry this
# string in its ``notes`` column so a downstream reader sees the value is
# computed, not measured.
NOTES_TAG_PROP_ENERGY: str = (
    "energy_per_query: proportionally allocated from block tracker"
)


def make_run_id(
    config_hash: str,
    ts: Any = None,
    repetition: int | None = None,
) -> tuple[str, str, str]:
    """Generate a UTC run identifier for a single (config, repetition) run.

    The run_id format is ``{YYYYMMDDTHHMMSSZ}_{config_hash}`` with an optional
    ``_r{repetition}`` suffix. Phase C used the bare form (1 indexing + 13
    queries share one run_id). Phase D adds the ``_r{rep}`` suffix to every
    *query* run_id while keeping the *indexing* run_id bare — the indexing
    run_id is therefore a strict prefix of the matching query run_ids,
    making cost↔repetition joins straightforward in Phase E.

    Args:
        config_hash: First 8 hex chars of the config SHA1 (see
            :class:`src.pipeline.RAGConfig`).
        ts: Optional ``datetime`` to use as the timestamp anchor. Defaults
            to ``datetime.now(timezone.utc)``. Pass an explicit value to
            keep all rows of one notebook batch joinable.
        repetition: Optional 1-indexed repetition number. ``None`` produces
            the bare indexing-style id. ``1``, ``2``, ``3`` produce the
            query-style id with ``_r1`` / ``_r2`` / ``_r3``.

    Returns:
        Tuple of ``(ts_iso, ts_compact, run_id)`` for direct use in the §9
        ``timestamp`` and ``run_id`` columns.
    """
    from datetime import datetime, timezone

    ts_dt = ts if ts is not None else datetime.now(timezone.utc)
    ts_iso = ts_dt.isoformat(timespec="seconds")
    ts_compact = ts_dt.strftime("%Y%m%dT%H%M%SZ")
    base = f"{ts_compact}_{config_hash}"
    run_id = base if repetition is None else f"{base}_r{int(repetition)}"
    return ts_iso, ts_compact, run_id


def assert_indexing_schema(df: Any) -> None:
    """Assert that ``df.columns`` matches §9 ``indexing_log.csv`` exactly.

    Compares by name AND order so a typo or column reorder is caught before
    value-level asserts produce a confusing ``KeyError`` trace.
    """
    got = list(df.columns)
    if got != EXPECTED_INDEXING_COLS:
        raise AssertionError(
            "indexing_log.csv columns drifted from §9 schema:\n"
            f"  expected: {EXPECTED_INDEXING_COLS}\n"
            f"  got:      {got}"
        )


def assert_query_schema(df: Any) -> None:
    """Assert that ``df.columns`` matches §9 ``query_log.csv`` exactly."""
    got = list(df.columns)
    if got != EXPECTED_QUERY_COLS:
        raise AssertionError(
            "query_log.csv columns drifted from §9 schema:\n"
            f"  expected: {EXPECTED_QUERY_COLS}\n"
            f"  got:      {got}"
        )


def allocate_block_energy_proportionally(
    rows: list[dict[str, Any]],
    block_energy_wh: float,
    block_co2_g: float,
    *,
    weight_key: str = "total_latency_ms",
    notes_tag: str = NOTES_TAG_PROP_ENERGY,
) -> None:
    """Distribute a CodeCarbon block measurement across its constituent rows.

    Implements Phase C decision Q2: a single :class:`OfflineEmissionsTracker`
    is started before the query loop and stopped after it; per-query
    energy/CO2 is then allocated by ``per_row_weight / sum(weights)``.
    Default weight is ``total_latency_ms`` because the LLM call is ~98% of
    per-query wall-clock at the M4 baseline and the dominant driver of CPU
    power draw during the block.

    Mutates each row in place, setting ``energy_wh_per_query``,
    ``co2_g_per_query``, and ``notes`` (idempotent if called twice with the
    same inputs). Raises :class:`AssertionError` if the sum of weights is
    non-positive (indicates an empty or no-op query loop).

    Args:
        rows: List of dicts that will become rows in ``query_log.csv``;
            each must already contain the ``weight_key`` field.
        block_energy_wh: Total energy (Wh) reported by the CodeCarbon
            tracker over the whole block.
        block_co2_g: Total CO2-equivalent (g) reported by the tracker.
        weight_key: Field used as the proportional weight. Defaults to
            ``"total_latency_ms"``.
        notes_tag: String written to the ``notes`` column of every row.
    """
    if not rows:
        raise AssertionError(
            "allocate_block_energy_proportionally: rows is empty — query "
            "loop did no work?"
        )
    total_weight = sum(float(r[weight_key]) for r in rows)
    if total_weight <= 0:
        raise AssertionError(
            f"allocate_block_energy_proportionally: sum({weight_key}) is "
            f"{total_weight!r} — non-positive weight."
        )
    for r in rows:
        w = float(r[weight_key]) / total_weight
        r["energy_wh_per_query"] = float(block_energy_wh) * w
        r["co2_g_per_query"]     = float(block_co2_g)    * w
        r["notes"]               = notes_tag


def assert_query_sanity(
    qry_this: Any,
    idx_this: Any,
    *,
    top_k: int,
    chunk_size: int,
    expected_query_count: int,
    expected_indexing_count: int = 1,
    notes_tag: str = NOTES_TAG_PROP_ENERGY,
    retrieved_token_slack: float = 1.10,
) -> None:
    """Run sanity checks A–H on a (already-filtered) pair of dataframes.

    The caller is responsible for slicing ``indexing_log.csv`` and
    ``query_log.csv`` down to the rows being checked (typically: filter by
    ``run_id`` for one configuration). Phase D will call this once per
    configuration after appending its rows.

    Checks (failing message identifies the letter):

    - **A** Row counts match ``expected_indexing_count`` and ``expected_query_count``.
    - **B** Every query row's ``config_hash`` matches the indexing row's.
    - **C** ``generation_time_ms > retrieval_time_ms`` for every row.
    - **D** ``retrieved_token_count <= top_k * chunk_size * retrieved_token_slack``.
    - **E** Indexing ``energy_wh > 0`` and ``co2_g > 0``.
    - **F** ``200 < peak_ram_mb < 8000``.
    - **G** Every ``answer_text`` is non-empty.
    - **H** Every query row's ``notes`` equals ``notes_tag``.

    Args:
        qry_this: Filtered ``query_log.csv`` rows for this run/config.
        idx_this: Filtered ``indexing_log.csv`` rows for this run/config
            (typically exactly one row).
        top_k: ``RAGConfig.top_k`` for the run being checked.
        chunk_size: ``RAGConfig.chunk_size`` for the run being checked.
        expected_query_count: Expected ``len(qry_this)``. Phase C: 13.
            Phase D Exp1: 39 per config (13 questions × 3 reps).
        expected_indexing_count: Expected ``len(idx_this)``. Default 1.
        notes_tag: Expected value of every query row's ``notes`` column.
        retrieved_token_slack: Multiplier applied to the
            ``top_k * chunk_size`` ceiling (Phase C Q6: 10% slack for RCTS
            occasionally overshooting).
    """
    # A — row counts
    if len(idx_this) != expected_indexing_count:
        raise AssertionError(
            f"FAIL (A): expected {expected_indexing_count} indexing row(s), "
            f"got {len(idx_this)}"
        )
    if len(qry_this) != expected_query_count:
        raise AssertionError(
            f"FAIL (A): expected {expected_query_count} query row(s), "
            f"got {len(qry_this)}"
        )

    # B — config_hash linkage
    expected_hash = idx_this["config_hash"].iloc[0]
    bad_hash = qry_this[qry_this["config_hash"] != expected_hash]
    if not bad_hash.empty:
        raise AssertionError(
            f"FAIL (B): query rows have config_hash != {expected_hash!r}: "
            f"{bad_hash['question_id'].tolist()}"
        )

    # C — generation dominates retrieval
    bad_gen = qry_this[
        qry_this["generation_time_ms"] <= qry_this["retrieval_time_ms"]
    ]
    if not bad_gen.empty:
        raise AssertionError(
            "FAIL (C): generation_time_ms must be > retrieval_time_ms "
            "for every row.\n"
            f"{bad_gen[['question_id','retrieval_time_ms','generation_time_ms']]}"
        )

    # D — retrieved_token_count ceiling (Q6 slack)
    ceiling = float(top_k) * float(chunk_size) * float(retrieved_token_slack)
    bad_tok = qry_this[qry_this["retrieved_token_count"] > ceiling]
    if not bad_tok.empty:
        raise AssertionError(
            f"FAIL (D): retrieved_token_count exceeds "
            f"k*chunk_size*{retrieved_token_slack:.2f} = {ceiling:.0f}.\n"
            f"{bad_tok[['question_id','top_k','retrieved_token_count']]}"
        )

    # E — non-zero energy/CO2 on the indexing row
    if float(idx_this["energy_wh"].iloc[0]) <= 0:
        raise AssertionError(
            "FAIL (E): indexing energy_wh is 0 — CodeCarbon misconfigured?"
        )
    if float(idx_this["co2_g"].iloc[0]) <= 0:
        raise AssertionError(
            "FAIL (E): indexing co2_g is 0 — CodeCarbon misconfigured?"
        )

    # F — plausible peak RAM
    peak = float(idx_this["peak_ram_mb"].iloc[0])
    if not (200 < peak < 8000):
        raise AssertionError(
            f"FAIL (F): peak_ram_mb={peak:.1f} MB outside plausible "
            f"200–8000 MB range"
        )

    # G — non-empty answers
    empty = qry_this[qry_this["answer_text"].astype(str).str.strip() == ""]
    if not empty.empty:
        raise AssertionError(
            f"FAIL (G): empty answer_text for {empty['question_id'].tolist()}"
        )

    # H — Q2 notes tag present on every query row
    missing_tag = qry_this[qry_this["notes"] != notes_tag]
    if not missing_tag.empty:
        raise AssertionError(
            f"FAIL (H): notes tag missing on "
            f"{missing_tag['question_id'].tolist()}"
        )


__all__ = [
    "timing_context",
    "time_it",
    "PeakRAMSampler",
    "count_text_tokens",
    "count_retrieved_tokens",
    "set_global_seeds",
    # C.6 result-log helpers
    "EXPECTED_INDEXING_COLS",
    "EXPECTED_QUERY_COLS",
    "NOTES_TAG_PROP_ENERGY",
    "make_run_id",
    "assert_indexing_schema",
    "assert_query_schema",
    "allocate_block_energy_proportionally",
    "assert_query_sanity",
]
