"""Evaluation question loader for the SD-A-01-RAG corpus (Phase B.8 / Phase D).

Parses ``dataset/eval_questions.md`` into structured :class:`EvalQuestion`
objects. The markdown layout is fixed (see the source file): each question is
a ``## Question NN`` section with five ``**Field:**``-labelled blocks
delimited by ``---`` lines.

Multi-doc questions (currently only Q13) carry a ``source_docs`` tuple of
length > 1; single-doc questions carry length 1. The convenience
``source_doc`` field joins them with ``"+"`` for round-trip into CSV.

Public API:
    - :class:`EvalQuestion` — frozen dataclass with the parsed fields.
    - :func:`load_eval_questions` — parse the markdown file and return a list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.pdf_extraction import PDF_TO_SHORT_NAME

# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #

VALID_QUESTION_TYPES: frozenset[str] = frozenset(
    {"factoid", "numerical", "synthesis", "multi-doc"}
)

# Order matters for the parser — fields are extracted left-to-right and the
# next-field lookahead uses this list.
_FIELDS: tuple[str, ...] = (
    "Source document",
    "Question type",
    "Question",
    "Reference answer",
    "Source location",
)


@dataclass(frozen=True)
class EvalQuestion:
    """A single evaluation question with reference answer and provenance.

    Attributes:
        question_id: ``"Q01"`` … ``"Q13"`` (zero-padded to 2 digits).
        source_docs: Tuple of short doc names. Length 1 for single-doc
            questions, length 2 for the Q13 multi-doc case.
        source_doc: ``"+"``-joined ``source_docs`` (CSV-friendly convenience).
        question_type: One of :data:`VALID_QUESTION_TYPES`.
        question: Question text, with soft-wrap newlines collapsed to spaces.
        reference_answer: Reference answer text, ditto.
        source_location: Human-readable source location (page/section), ditto.
    """

    question_id: str
    source_docs: tuple[str, ...]
    source_doc: str
    question_type: str
    question: str
    reference_answer: str
    source_location: str


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


_RE_QUESTION_HEADER = re.compile(r"^## Question (\d+)\s*$", re.MULTILINE)
_RE_WS = re.compile(r"\s+")


def _collapse_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (including newlines) to single spaces.

    The markdown soft-wraps long fields across multiple lines for readability;
    we don't want those soft-wrap newlines bleeding into prompts or CSV.
    """
    return _RE_WS.sub(" ", text).strip()


def _extract_field(block: str, field: str) -> str:
    """Extract one ``**field:**`` value from a question block.

    The match terminates at the next ``**word:**`` label or end of block.

    Raises:
        ValueError: If the field is missing or empty.
    """
    # Lookahead: next field label OR end of string.
    pattern = re.compile(
        rf"\*\*{re.escape(field)}:\*\*\s*(.*?)(?=\n\*\*\w[\w ]*:\*\*|\Z)",
        re.DOTALL,
    )
    m = pattern.search(block)
    if m is None:
        raise ValueError(f"Field '**{field}:**' not found in block.")
    value = _collapse_whitespace(m.group(1))
    if not value:
        raise ValueError(f"Field '**{field}:**' is empty.")
    return value


def _resolve_source_docs(raw: str) -> tuple[str, ...]:
    """Map the ``Source document`` field value to short doc names.

    Single-doc: ``"01_BLOOM_Luccioni_2022.pdf"`` -> ``("01_BLOOM",)``.
    Multi-doc: ``"01_BLOOM_Luccioni_2022.pdf + 05_Greenpeace_..."`` ->
    ``("01_BLOOM", "05_Greenpeace")``.
    """
    parts = [p.strip() for p in raw.split("+")]
    short_names: list[str] = []
    for p in parts:
        if p not in PDF_TO_SHORT_NAME:
            raise ValueError(
                f"Unknown source document '{p}'. Expected one of "
                f"{sorted(PDF_TO_SHORT_NAME)}."
            )
        short_names.append(PDF_TO_SHORT_NAME[p])
    return tuple(short_names)


# --------------------------------------------------------------------------- #
# Public loader                                                               #
# --------------------------------------------------------------------------- #


def load_eval_questions(
    path: Path,
    expected_count: int = 13,
) -> list[EvalQuestion]:
    """Parse ``eval_questions.md`` into a list of :class:`EvalQuestion`.

    Args:
        path: Path to the markdown file (typically
            ``dataset/eval_questions.md``).
        expected_count: Hard assertion on the number of questions parsed.
            Defaults to 13 to catch silent file truncation.

    Returns:
        List of :class:`EvalQuestion`, ordered by ``question_id``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On any structural problem — wrong question count, missing
            field, unknown question type, unknown source document.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"eval_questions file not found: {path}")
    text = path.read_text(encoding="utf-8")

    # Find each question header position.
    headers = list(_RE_QUESTION_HEADER.finditer(text))
    if len(headers) != expected_count:
        raise ValueError(
            f"Expected {expected_count} questions in {path}, "
            f"found {len(headers)}."
        )

    questions: list[EvalQuestion] = []
    for i, m in enumerate(headers):
        q_num = int(m.group(1))
        # Slice the block: from end-of-header to start-of-next-header (or EOF).
        block_start = m.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[block_start:block_end]

        try:
            source_doc_raw = _extract_field(block, "Source document")
            q_type = _extract_field(block, "Question type")
            q_text = _extract_field(block, "Question")
            ref_answer = _extract_field(block, "Reference answer")
            src_loc = _extract_field(block, "Source location")
        except ValueError as e:
            raise ValueError(f"Question {q_num}: {e}") from e

        if q_type not in VALID_QUESTION_TYPES:
            raise ValueError(
                f"Question {q_num}: unknown question_type {q_type!r}. "
                f"Expected one of {sorted(VALID_QUESTION_TYPES)}."
            )

        source_docs = _resolve_source_docs(source_doc_raw)
        questions.append(
            EvalQuestion(
                question_id=f"Q{q_num:02d}",
                source_docs=source_docs,
                source_doc="+".join(source_docs),
                question_type=q_type,
                question=q_text,
                reference_answer=ref_answer,
                source_location=src_loc,
            )
        )

    # Order by question_id (string comparison is stable because IDs are
    # zero-padded to 2 digits).
    questions.sort(key=lambda q: q.question_id)
    return questions


__all__ = [
    "VALID_QUESTION_TYPES",
    "EvalQuestion",
    "load_eval_questions",
]
