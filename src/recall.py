"""Recall@k computation against page-anchored ground truth (Phase E.1 / E.2).

See MASTER_PLAN.md §11 (Phase E) and PROGRESS_LOG.md Phase E / Batch A entry.

This module implements the retroactive ground-truth layer that the §11 E.1
"ground truth" paragraph said was supposed to be added in Phase B but wasn't:

1. Parse the human-readable ``Source location`` field on each evaluation
   question into a per-source-document set of page numbers.
2. For each corpus PDF, re-derive the character offset of every page in
   ``corpus_text/<doc>.txt`` by extracting per-page text with pypdf, cleaning
   each page identically to ``src.pdf_extraction.clean_extracted_text``, then
   locating each page's content in the on-disk corpus text.
3. For each (config_hash, question) pair, intersect the gold character ranges
   with each chunk's ``[start_char, end_char)`` interval to produce the
   ``gold_chunk_ids`` set.
4. For each (config_hash, top_k, question) cell in ``query_log.csv``, compute
   ``hit = bool(gold_chunk_ids & retrieved_chunk_ids)``. For Q13 (multi-doc),
   additionally compute ``strict_hit`` (chunks from BOTH source docs are
   retrieved) and ``lenient_hit`` (chunks from AT LEAST ONE source doc are
   retrieved) — see §11 E.2.

Whitespace robustness:
    pypdf extraction is largely deterministic on a fixed machine but small
    differences across pypdf versions (line breaks landing in different
    positions, occasional letter-spacing artifacts, ``"L IGOZAT"`` vs
    ``"LIGOZAT"``) can produce per-page text that differs from
    ``corpus_text/<doc>.txt`` even though both are pypdf output.

    The matcher therefore searches in a whitespace-stripped *normalized*
    form: every non-whitespace character of the corpus is concatenated into
    a single string, with a parallel index mapping back to original char
    offsets. Fingerprints from each cleaned page are likewise stripped of
    whitespace and located in the normalized corpus. The recovered
    normalized position is mapped back to its original char offset for the
    PageOffset record. This absorbs all observed pypdf-version drift.

    On the project's primary measurement machine (the user's MacBook Air
    M4), pypdf produces text byte-identical to ``corpus_text/`` and the
    matcher's first fingerprint hits on the very first try; the
    normalization machinery is only load-bearing on cross-machine re-runs.

Public API:
    - :class:`PageOffset` — per-page char range in ``corpus_text/<doc>.txt``.
    - :class:`GoldRange` — per-doc, per-question gold character range.
    - :class:`RecallRow` — schema for ``results/recall_log.csv``.
    - :func:`parse_source_pages` — parse the ``Source location`` field.
    - :func:`build_page_offsets` — page-to-char-range table for one doc.
    - :func:`gold_ranges_for_question` — list of GoldRange for one question.
    - :func:`gold_chunk_ids` — chunk_ids overlapping any gold range.
    - :func:`compute_hit` — boolean overlap between retrieved and gold ids.
    - :func:`compute_q13_hits` — strict + lenient flags for Q13 multi-doc.
    - :func:`build_recall_rows` — produce all rows for ``recall_log.csv``.

Schema for ``results/recall_log.csv``:
    config_hash, top_k, question_id, question_type, source_doc,
    n_gold_chunks, gold_chunk_ids, retrieved_chunk_ids, hit, strict_hit,
    lenient_hit, notes

    ``strict_hit`` and ``lenient_hit`` are non-empty only for Q13; for
    every other question they are written as the empty string. ``hit`` is
    always set: for Q13 it equals ``lenient_hit`` (the project decision is
    that single-doc recall@k aggregates use the lenient definition; the
    strict-vs-lenient gap is a separate reported number).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from src.chunking import Chunk
from src.eval_questions import EvalQuestion
from src.pdf_extraction import (
    PDF_TO_SHORT_NAME,
    clean_extracted_text,
)

# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #

#: §11 schema — recall_log.csv. Order matters; ``assert_recall_schema``
#: checks both name and order.
EXPECTED_RECALL_COLS: list[str] = [
    "config_hash",
    "top_k",
    "question_id",
    "question_type",
    "source_doc",
    "n_gold_chunks",
    "gold_chunk_ids",
    "retrieved_chunk_ids",
    "hit",
    "strict_hit",
    "lenient_hit",
    "notes",
]

#: Multi-doc question id — only Q13 has strict / lenient variants.
Q13_ID: str = "Q13"

#: ``;`` separator inherited from Phase B/C convention for list-typed CSV
#: columns (``retrieved_chunk_ids``, ``retrieved_scores``).
LIST_SEP: str = ";"


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PageOffset:
    """Character range of one PDF page in ``corpus_text/<doc>.txt``.

    Attributes:
        page: 1-indexed page number (matches ``Source location`` notation).
        start_char: Inclusive start offset, or ``None`` if the page extracted
            empty (image-only PDF page).
        end_char: Exclusive end offset, or ``None`` for empty pages. The end
            is the start of the next non-empty page (or the corpus length
            for the last page).
    """

    page: int
    start_char: int | None
    end_char: int | None


@dataclass(frozen=True)
class GoldRange:
    """One gold character range for one (question, source_doc) pair."""

    question_id: str
    source_doc: str
    page: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class RecallRow:
    """One row of ``results/recall_log.csv``."""

    config_hash: str
    top_k: int
    question_id: str
    question_type: str
    source_doc: str
    n_gold_chunks: int
    gold_chunk_ids: str
    retrieved_chunk_ids: str
    hit: bool
    strict_hit: str   # "" | "True" | "False"  (only Q13 sets this)
    lenient_hit: str  # "" | "True" | "False"  (only Q13 sets this)
    notes: str


# --------------------------------------------------------------------------- #
# Source-location parsing                                                     #
# --------------------------------------------------------------------------- #


# Match a number, a hyphen-style range, or a comma/and-joined sequence of
# either. Captures the entire token after ``page(s)``. Hyphens covered:
# ASCII ``-``, en-dash ``–`` (U+2013), em-dash ``—`` (U+2014).
_PAGE_TOKEN_RE = re.compile(
    r"(?:^|\W)pages?\s+("
    r"\d+(?:\s*[\-–—]\s*\d+)?"
    r"(?:\s*(?:,|and)\s*\d+(?:\s*[\-–—]\s*\d+)?)*"
    r")",
    re.IGNORECASE,
)

_RANGE_PIECE_RE = re.compile(
    r"(\d+)\s*[\-–—]\s*(\d+)"
)


def _expand_page_token(token: str) -> set[int]:
    """Expand a parsed page token (e.g. ``"9–10, 12 and 14-15"``) to a set of ints."""
    pages: set[int] = set()
    # Split on commas and the word "and" (case-insensitive). Whitespace
    # around the splitter is trimmed.
    parts = [p.strip() for p in re.split(r",\s*|\s+and\s+", token) if p.strip()]
    for part in parts:
        m = _RANGE_PIECE_RE.fullmatch(part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            pages.update(range(a, b + 1))
            continue
        m2 = re.fullmatch(r"\d+", part)
        if m2:
            pages.add(int(part))
    return pages


# Q13 source_location is the only multi-doc case in the eval set. It looks
# like:
#   "BLOOM: Sections 4.1 and 4.3 (pages 4–7); Greenpeace: Introduction and
#   Section 'Energy Demand from AI Chipmaking' (pages 10–11, 16–22)"
# The parser splits on ";" and looks for the doc-name keyword at the start
# of each segment.
_DOC_KEYWORD_TO_SHORT: dict[str, str] = {
    "bloom": "01_BLOOM",
    "iea": "02_IEA",
    "google": "03_Google",
    "epri": "04_EPRI",
    "greenpeace": "05_Greenpeace",
}


def parse_source_pages(
    question: EvalQuestion,
) -> dict[str, set[int]]:
    """Parse ``question.source_location`` into a per-doc set of page numbers.

    Single-doc questions: returns ``{question.source_docs[0]: pages}``.
    Multi-doc questions (Q13): segments delimited by ``;`` are inspected for
    a doc-name keyword (BLOOM / IEA / Google / EPRI / Greenpeace) and pages
    are attributed to the matching short name. Segments without a keyword
    are attributed to the question's first source_doc as a safe default.

    Args:
        question: An :class:`src.eval_questions.EvalQuestion`.

    Returns:
        Mapping from short doc name (``"01_BLOOM"`` etc.) to the set of
        1-indexed page numbers cited in ``Source location``.

    Raises:
        ValueError: If no page numbers can be parsed for any source doc.
    """
    src_loc = question.source_location
    pages_by_doc: dict[str, set[int]] = {d: set() for d in question.source_docs}

    if len(question.source_docs) == 1:
        # Single-doc: every page hit goes to the one source doc.
        target = question.source_docs[0]
        for m in _PAGE_TOKEN_RE.finditer(src_loc):
            pages_by_doc[target].update(_expand_page_token(m.group(1)))
    else:
        # Multi-doc (Q13). Segment on ";" — each segment owns its pages.
        segments = [s.strip() for s in src_loc.split(";") if s.strip()]
        if not segments:
            segments = [src_loc]
        for seg in segments:
            # Identify the doc this segment refers to.
            seg_lower = seg.lower()
            owner: str | None = None
            for kw, short in _DOC_KEYWORD_TO_SHORT.items():
                if kw in seg_lower and short in question.source_docs:
                    owner = short
                    break
            if owner is None:
                # Fallback: attribute to the first listed source_doc so we
                # still surface SOMETHING rather than dropping the segment.
                owner = question.source_docs[0]
            for m in _PAGE_TOKEN_RE.finditer(seg):
                pages_by_doc[owner].update(_expand_page_token(m.group(1)))

    if not any(pages_by_doc.values()):
        raise ValueError(
            f"parse_source_pages: no pages parsed for {question.question_id} "
            f"from source_location={src_loc!r}"
        )
    return pages_by_doc


# --------------------------------------------------------------------------- #
# Page-offset detection                                                       #
# --------------------------------------------------------------------------- #


def _normalize_for_match(text: str) -> tuple[str, list[int]]:
    """Strip all whitespace from ``text`` and return a parallel offset map.

    Returns:
        A tuple ``(norm, mapping)`` where ``norm`` is ``text`` with every
        whitespace character removed, and ``mapping[i]`` is the offset in
        the *original* ``text`` of the character that became ``norm[i]``.
        This lets us search in the normalized space and project the match
        back to the original char offset.
    """
    chars: list[str] = []
    mapping: list[int] = []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            mapping.append(i)
    return "".join(chars), mapping


def _windowed_fingerprints(norm_body: str) -> list[tuple[int, str]]:
    """Yield ``(offset, fingerprint)`` windows from the normalized page body.

    Each window's ``offset`` is its starting offset within ``norm_body``; the
    matcher uses this to project a fingerprint match back to the page's true
    start. Windows are ordered so that the first ones tried are
    (a) longest (highest uniqueness; immune to short-fingerprint aliasing
    against earlier text) and (b) sourced from the page body interior, where
    pypdf-version drift is less likely than at the header/footer.

    We always include several body-start windows too, because the header
    line (e.g., a recurring journal banner) is the most reliable signal
    when the PDF really does start a new page there.

    Sizes are in *normalized* characters (whitespace already stripped).
    """
    n = len(norm_body)
    out: list[tuple[int, str]] = []

    # 1. Mid-body windows — most distinctive narrative content; longest first.
    if n >= 400:
        mid = n // 2
        out.append((mid, norm_body[mid : mid + 200]))
    if n >= 240:
        mid = n // 2
        out.append((mid, norm_body[mid : mid + 120]))

    # 2. Late-body window — skips footer / page-number line.
    if n >= 700:
        late = n - 300
        out.append((late, norm_body[late : late + 200]))

    # 3. Skip-header window — first 100 chars are header, the next 200 are
    #    page-specific narrative.
    if n >= 320:
        out.append((100, norm_body[100 : 300]))

    # 4. Start-of-page windows — long → short. These catch pages whose
    #    header IS the right anchor (e.g., short cover pages, single-line
    #    pages) and pages where mid-body content is also drift-affected.
    for size in (300, 200, 120, 80, 50, 30, 20, 15, 10):
        if size <= n:
            out.append((0, norm_body[:size]))

    # 5. Last resort for tiny pages: the whole body. (Will already be in (4)
    #    if n<= 10, but kept as an explicit final fallback.)
    if n < 10 and n > 0:
        out.append((0, norm_body))

    # De-duplicate by fingerprint string while preserving order.
    seen: set[str] = set()
    deduped: list[tuple[int, str]] = []
    for offset, fp in out:
        if fp and fp not in seen:
            seen.add(fp)
            deduped.append((offset, fp))
    return deduped


def _find_page_start_norm(
    windows: list[tuple[int, str]],
    norm_corpus: str,
    cursor_norm: int,
) -> int:
    """Return the page's start position in the *normalized* corpus, or ``-1``.

    Each candidate window is a ``(offset, fingerprint)`` pair: the
    fingerprint is what we search for, and ``offset`` is how many chars into
    the page the fingerprint starts. The match position implies a page
    start at ``match_pos - offset``. We require that page-start to be
    ``>= cursor_norm`` (forward-only progression).

    Tries each window in the supplied order; returns the first valid match
    position. Longer / more-distinctive windows come first (see
    :func:`_windowed_fingerprints`), so a generic short fingerprint can't
    pre-empt the correct page boundary.
    """
    for offset, fp in windows:
        # Earliest acceptable match position so that page_start >= cursor_norm.
        min_pos = cursor_norm + offset
        pos = norm_corpus.find(fp, min_pos)
        if pos >= 0:
            page_start = pos - offset
            if page_start >= cursor_norm:
                return page_start
    return -1


def build_page_offsets(
    pdf_path: Path,
    corpus_text: str,
) -> list[PageOffset]:
    """Return per-page ``PageOffset`` records for ``corpus_text/<doc>.txt``.

    Each PDF page is extracted with pypdf and cleaned identically to
    ``src.pdf_extraction.clean_extracted_text``; the resulting cleaned page
    text is then located in ``corpus_text`` using a forward-only cursor
    (page i+1 is searched from page i's start). The end_char of each page
    is the start_char of the next non-empty page; the last non-empty page
    ends at ``len(corpus_text)``.

    Args:
        pdf_path: Path to the source PDF.
        corpus_text: The on-disk content of the matching ``corpus_text/<doc>.txt``.

    Returns:
        List of :class:`PageOffset`, one per PDF page, in page order.
        Image-only / empty pages have ``start_char = end_char = None``.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        RuntimeError: If a non-empty page cannot be located in ``corpus_text``
            even after the fingerprint fallbacks. The error message includes
            the page number and the first 80 chars of the cleaned page body
            so the failure is debuggable on first read.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"build_page_offsets: PDF not found: {pdf_path}")

    # Build a parallel normalized view of the corpus once.
    norm_corpus, norm_to_orig = _normalize_for_match(corpus_text)

    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    starts: list[int | None] = [None] * n_pages
    cursor_norm = 0
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = clean_extracted_text(raw).strip()
        if not cleaned:
            continue
        # Normalize the cleaned page body the same way as the corpus.
        norm_body, _ = _normalize_for_match(cleaned)
        if not norm_body:
            continue
        windows = _windowed_fingerprints(norm_body)
        pos_norm = _find_page_start_norm(windows, norm_corpus, cursor_norm)
        if pos_norm < 0:
            preview = cleaned[:80].replace("\n", "\\n")
            raise RuntimeError(
                f"build_page_offsets: could not locate page {i + 1} of "
                f"{pdf_path.name} in corpus_text (norm cursor={cursor_norm}). "
                f"Page preview: {preview!r}..."
            )
        # Project the normalized position back to an original char offset.
        starts[i] = int(norm_to_orig[pos_norm])
        # Advance the normalized cursor by 1 so the next page's search skips
        # the just-found start but can still find an immediately-following
        # page (e.g., back-to-back short pages with similar headers).
        cursor_norm = pos_norm + 1

    # Compute end_char: each page ends where the NEXT non-empty page starts;
    # the last non-empty page ends at len(corpus_text).
    page_offsets: list[PageOffset] = []
    for i, start in enumerate(starts):
        if start is None:
            page_offsets.append(PageOffset(page=i + 1, start_char=None, end_char=None))
            continue
        # Find the next non-None start.
        end: int = len(corpus_text)
        for j in range(i + 1, n_pages):
            if starts[j] is not None:
                end = starts[j]
                break
        page_offsets.append(
            PageOffset(page=i + 1, start_char=int(start), end_char=int(end))
        )
    return page_offsets


# --------------------------------------------------------------------------- #
# Gold ranges + chunk overlap                                                 #
# --------------------------------------------------------------------------- #


def gold_ranges_for_question(
    question: EvalQuestion,
    page_offsets_by_doc: dict[str, list[PageOffset]],
) -> list[GoldRange]:
    """Compute the gold character ranges for one question.

    Page numbers are taken from :func:`parse_source_pages`. For each page,
    the (start_char, end_char) range comes from ``page_offsets_by_doc``.
    Pages that resolve to an empty / unmappable page (``start_char is None``)
    are skipped silently — the cleaning policy occasionally produces empty
    pages for cover sheets and image-only pages, and those legitimately
    have no narrative content to retrieve.

    Args:
        question: The evaluation question.
        page_offsets_by_doc: Per-doc list of :class:`PageOffset`, keyed by
            short doc name.

    Returns:
        List of :class:`GoldRange`, one per (question, doc, page) triple
        that resolved to a non-empty page.

    Raises:
        KeyError: If a source doc has no page-offset table.
    """
    pages_by_doc = parse_source_pages(question)
    out: list[GoldRange] = []
    for doc, pages in pages_by_doc.items():
        if doc not in page_offsets_by_doc:
            raise KeyError(
                f"gold_ranges_for_question: no page_offsets for doc {doc!r}"
            )
        offsets = page_offsets_by_doc[doc]
        page_to_offset = {po.page: po for po in offsets}
        for p in sorted(pages):
            po = page_to_offset.get(p)
            if po is None:
                # Page number out of range for this document — surface as a
                # data-quality warning by raising; this catches typos in
                # eval_questions.md early.
                raise ValueError(
                    f"gold_ranges_for_question: {question.question_id} cites "
                    f"{doc} page {p} but the PDF has {len(offsets)} pages"
                )
            if po.start_char is None or po.end_char is None:
                # Empty / image-only page — skip silently.
                continue
            out.append(
                GoldRange(
                    question_id=question.question_id,
                    source_doc=doc,
                    page=p,
                    start_char=po.start_char,
                    end_char=po.end_char,
                )
            )
    return out


def gold_chunk_ids(
    gold_ranges: list[GoldRange],
    chunks_by_doc: dict[str, list[Chunk]],
) -> set[str]:
    """Return the set of chunk_ids that overlap any gold character range.

    Overlap is the standard half-open interval test::

        max(chunk.start, gold.start) < min(chunk.end, gold.end)

    Per-doc chunks are restricted to the gold range's source_doc, so a
    BLOOM gold range never matches a Greenpeace chunk (relevant for Q13).

    Args:
        gold_ranges: Output of :func:`gold_ranges_for_question`.
        chunks_by_doc: Mapping from short doc name to its full chunk list
            (typically grouped from a ``chunks.parquet`` load).

    Returns:
        Set of chunk_ids hitting at least one gold range. May be empty if
        every gold range falls outside any chunk (e.g., page extracted
        empty in this config).
    """
    hits: set[str] = set()
    for gr in gold_ranges:
        candidates = chunks_by_doc.get(gr.source_doc, [])
        for c in candidates:
            if max(c.start_char, gr.start_char) < min(c.end_char, gr.end_char):
                hits.add(c.chunk_id)
    return hits


# --------------------------------------------------------------------------- #
# Recall computation                                                          #
# --------------------------------------------------------------------------- #


def parse_retrieved(retrieved_chunk_ids: str) -> list[str]:
    """Parse the ``;``-separated retrieved_chunk_ids field from query_log.csv."""
    if not retrieved_chunk_ids:
        return []
    return [s for s in retrieved_chunk_ids.split(LIST_SEP) if s]


def compute_hit(retrieved: Iterable[str], gold: set[str]) -> bool:
    """Boolean: at least one retrieved chunk_id is in the gold set."""
    if not gold:
        return False
    return any(r in gold for r in retrieved)


def compute_q13_hits(
    retrieved: Iterable[str],
    gold_by_doc: dict[str, set[str]],
) -> tuple[bool, bool]:
    """Compute ``(strict_hit, lenient_hit)`` for the Q13 multi-doc question.

    Strict: top-k contains chunk_ids from BOTH source documents (i.e., the
    intersection of retrieved with each per-doc gold set is non-empty for
    every doc).

    Lenient: top-k contains chunk_ids from AT LEAST ONE source document
    (i.e., the union of per-doc gold sets has a non-empty intersection
    with retrieved).

    Args:
        retrieved: The retrieved chunk_ids for this query.
        gold_by_doc: Mapping from source_doc short name to its gold
            chunk_id set, restricted to that doc.

    Returns:
        ``(strict, lenient)`` booleans. If ``gold_by_doc`` has only one
        non-empty doc gold set, strict and lenient are equal — but that
        case shouldn't arise for Q13 in practice.
    """
    retrieved_set = set(retrieved)
    per_doc_hit = [bool(retrieved_set & g) for g in gold_by_doc.values()]
    if not per_doc_hit:
        return False, False
    return (all(per_doc_hit), any(per_doc_hit))


# --------------------------------------------------------------------------- #
# Recall log builder                                                          #
# --------------------------------------------------------------------------- #


def build_recall_rows(
    queries: Iterable[dict],
    questions_by_id: dict[str, EvalQuestion],
    chunks_by_config: dict[str, dict[str, list[Chunk]]],
    page_offsets_by_doc: dict[str, list[PageOffset]],
) -> list[RecallRow]:
    """Compute one :class:`RecallRow` per (config_hash, top_k, question_id) cell.

    The caller is responsible for de-duplicating ``queries`` to one row per
    (config_hash, top_k, question_id) before calling — repetitions are
    byte-identical under temperature=0 + seed=42 (§13.5), so the dedup is a
    methodological clarification, not a measurement loss. This is the
    "locked decision" recorded in the Phase E / Batch A task brief.

    Args:
        queries: Iterable of dicts with at least ``config_hash``, ``top_k``,
            ``question_id``, ``retrieved_chunk_ids``. Typically
            ``df.to_dict(orient="records")`` over the deduplicated frame.
        questions_by_id: Mapping ``Q01 -> EvalQuestion``.
        chunks_by_config: Two-level mapping
            ``config_hash -> source_doc -> list[Chunk]``.
        page_offsets_by_doc: Mapping ``source_doc -> list[PageOffset]``.

    Returns:
        List of :class:`RecallRow` in input order.
    """
    rows: list[RecallRow] = []
    for q in queries:
        cfg_hash = str(q["config_hash"])
        top_k = int(q["top_k"])
        qid = str(q["question_id"])
        retrieved_str = str(q.get("retrieved_chunk_ids", "") or "")
        retrieved_list = parse_retrieved(retrieved_str)
        question = questions_by_id[qid]

        chunks_by_doc = chunks_by_config[cfg_hash]
        gold_ranges = gold_ranges_for_question(question, page_offsets_by_doc)
        gold_set = gold_chunk_ids(gold_ranges, chunks_by_doc)

        # For Q13 we need per-doc gold sets to compute strict/lenient hits.
        if qid == Q13_ID:
            gold_by_doc: dict[str, set[str]] = {}
            for doc in question.source_docs:
                doc_ranges = [gr for gr in gold_ranges if gr.source_doc == doc]
                gold_by_doc[doc] = gold_chunk_ids(doc_ranges, chunks_by_doc)
            strict, lenient = compute_q13_hits(retrieved_list, gold_by_doc)
            hit = lenient
            strict_str = str(bool(strict))
            lenient_str = str(bool(lenient))
            note = (
                "Q13 multi-doc: hit=lenient_hit (§11 E.2). "
                "strict_hit requires chunks from both source docs."
            )
        else:
            hit = compute_hit(retrieved_list, gold_set)
            strict_str = ""
            lenient_str = ""
            note = ""

        rows.append(
            RecallRow(
                config_hash=cfg_hash,
                top_k=top_k,
                question_id=qid,
                question_type=question.question_type,
                source_doc=question.source_doc,
                n_gold_chunks=len(gold_set),
                gold_chunk_ids=LIST_SEP.join(sorted(gold_set)),
                retrieved_chunk_ids=retrieved_str,
                hit=bool(hit),
                strict_hit=strict_str,
                lenient_hit=lenient_str,
                notes=note,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Schema lock                                                                 #
# --------------------------------------------------------------------------- #


def assert_recall_schema(df: object) -> None:
    """Assert that ``df.columns`` matches :data:`EXPECTED_RECALL_COLS` exactly.

    Mirrors :func:`src.metrics.assert_query_schema` — name AND order. Catches
    a column rename or reorder before downstream value asserts produce a
    confusing ``KeyError``.
    """
    got = list(getattr(df, "columns"))
    if got != EXPECTED_RECALL_COLS:
        raise AssertionError(
            "recall_log.csv columns drifted from §11 schema:\n"
            f"  expected: {EXPECTED_RECALL_COLS}\n"
            f"  got:      {got}"
        )


# --------------------------------------------------------------------------- #
# Convenience                                                                 #
# --------------------------------------------------------------------------- #


def recall_row_dict(row: RecallRow) -> dict:
    """Return the row as a plain dict, suitable for ``pd.DataFrame``."""
    return asdict(row)


__all__ = [
    "EXPECTED_RECALL_COLS",
    "Q13_ID",
    "LIST_SEP",
    "PageOffset",
    "GoldRange",
    "RecallRow",
    "parse_source_pages",
    "build_page_offsets",
    "gold_ranges_for_question",
    "gold_chunk_ids",
    "parse_retrieved",
    "compute_hit",
    "compute_q13_hits",
    "build_recall_rows",
    "assert_recall_schema",
    "recall_row_dict",
    "PDF_TO_SHORT_NAME",
]
