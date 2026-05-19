"""PDF -> plain-text extraction for the SD-A-01-RAG corpus.

This module is the Phase B.1 step of the project (see MASTER_PLAN.md §8). It
reads each of the 5 corpus PDFs in ``dataset/pdfs/`` with ``pypdf``, applies a
deliberately conservative cleaning pass, writes the result to
``corpus_text/<short_name>.txt``, and sanity-checks the resulting word counts
against ``dataset/corpus_stats.md`` within a configurable tolerance (default
±5%).

Cleaning policy (§8 B.1: "do NOT aggressively normalize"):
    1. Replace form-feed (``\\x0c``) — pypdf's page separator — with newline.
    2. Collapse runs of horizontal whitespace within a line to a single space.
    3. Strip per-line trailing whitespace.
    4. Collapse 3+ consecutive newlines to exactly two (paragraph break kept).

We deliberately do NOT de-hyphenate end-of-line word breaks, normalize quotes
or dashes, change case, or remove headers/footers. Any of those would shift
sentence boundaries before chunking and bias downstream retrieval.

Public functions:
    - :func:`extract_pdf_text` — extract and clean a single PDF.
    - :func:`clean_extracted_text` — the cleaning pass on its own.
    - :func:`extract_corpus` — run the full corpus, write .txt files, return stats.
    - :func:`verify_word_counts` — compare actual to expected within ±tolerance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pypdf import PdfReader

from src.metrics import time_it

# --------------------------------------------------------------------------- #
# Project-wide constants                                                      #
# --------------------------------------------------------------------------- #

#: Mapping from corpus PDF filenames to short output names.
#: Order matches ``dataset/corpus_stats.md`` and §6 of MASTER_PLAN.md.
PDF_TO_SHORT_NAME: dict[str, str] = {
    "01_BLOOM_Luccioni_2022.pdf": "01_BLOOM",
    "02_IEA_Energy_and_AI_2025.pdf": "02_IEA",
    "03_Google_Env_2025.pdf": "03_Google",
    "04_EPRI_Powering_Intelligence_2024.pdf": "04_EPRI",
    "05_Greenpeace_Chipping_Point_2025.pdf": "05_Greenpeace",
}

#: Reference word counts from ``dataset/corpus_stats.md`` (computed via
#: ``len(pypdf_extracted_text.split())``). Used by :func:`verify_word_counts`.
EXPECTED_WORD_COUNTS: dict[str, int] = {
    "01_BLOOM": 6_939,
    "02_IEA": 115_129,
    "03_Google": 56_660,
    "04_EPRI": 14_264,
    "05_Greenpeace": 7_025,
}


# --------------------------------------------------------------------------- #
# Cleaning                                                                    #
# --------------------------------------------------------------------------- #

# Pre-compiled regexes so the cleaning pass is cheap to call per PDF.
_RE_HORIZ_WS = re.compile(r"[ \t ]+")        # spaces, tabs, NBSPs
_RE_TRAIL_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_RE_3PLUS_NL = re.compile(r"\n{3,}")


def clean_extracted_text(text: str) -> str:
    """Apply minimal, sentence-boundary-preserving cleaning to extracted text.

    See module docstring for the policy. Idempotent.

    Args:
        text: Raw text as produced by ``pypdf``.

    Returns:
        Cleaned text.
    """
    # 1. Form-feed -> newline (pypdf page separator).
    text = text.replace("\x0c", "\n")
    # 2. Collapse runs of horizontal whitespace.
    text = _RE_HORIZ_WS.sub(" ", text)
    # 3. Strip per-line trailing whitespace.
    text = _RE_TRAIL_WS.sub("", text)
    # 4. Collapse 3+ newlines to exactly 2.
    text = _RE_3PLUS_NL.sub("\n\n", text)
    return text.strip() + "\n"


# --------------------------------------------------------------------------- #
# Per-PDF extraction                                                          #
# --------------------------------------------------------------------------- #


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a single PDF and return it cleaned.

    Pages are joined with ``"\\n"``. Pages with no extractable text contribute
    an empty string (rather than raising), which matches ``pypdf``'s default
    behavior for image-only pages.

    Args:
        pdf_path: Absolute or project-relative path to the PDF.

    Returns:
        Cleaned UTF-8 text. Always ends with a single trailing newline.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []
    for page in reader.pages:
        # extract_text() returns None for some image-only / empty pages.
        page_texts.append(page.extract_text() or "")
    return clean_extracted_text("\n".join(page_texts))


# --------------------------------------------------------------------------- #
# Corpus-level extraction                                                     #
# --------------------------------------------------------------------------- #


@time_it("pdf_extract")
def extract_corpus(
    pdf_dir: Path,
    out_dir: Path,
    pdf_to_short_name: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Extract every corpus PDF, write the .txt files, and return per-doc stats.

    Args:
        pdf_dir: Directory containing the source PDFs (e.g. ``dataset/pdfs/``).
        out_dir: Directory where cleaned ``.txt`` files are written
            (e.g. ``corpus_text/``). Created if it does not exist.
        pdf_to_short_name: Mapping from PDF filename to short output stem.
            Defaults to :data:`PDF_TO_SHORT_NAME`.

    Returns:
        Dict keyed by short name with per-doc stats::

            {
                "01_BLOOM": {
                    "pdf_path": Path,
                    "txt_path": Path,
                    "words": int,
                    "chars": int,
                },
                ...
            }

        ``words`` is computed as ``len(text.split())`` to match the recipe
        documented in ``dataset/corpus_stats.md``.

    Raises:
        FileNotFoundError: If any expected PDF is missing.
    """
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = pdf_to_short_name or PDF_TO_SHORT_NAME

    stats: dict[str, dict] = {}
    for pdf_name, short_name in mapping.items():
        pdf_path = pdf_dir / pdf_name
        txt_path = out_dir / f"{short_name}.txt"

        text = extract_pdf_text(pdf_path)
        txt_path.write_text(text, encoding="utf-8")

        stats[short_name] = {
            "pdf_path": pdf_path,
            "txt_path": txt_path,
            "words": len(text.split()),
            "chars": len(text),
        }
    return stats


# --------------------------------------------------------------------------- #
# Sanity check                                                                #
# --------------------------------------------------------------------------- #


def verify_word_counts(
    actual: dict[str, dict] | dict[str, int],
    expected: dict[str, int] | None = None,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Compare actual word counts against expected and assert they are close.

    Args:
        actual: Either the full stats dict from :func:`extract_corpus`
            (values are ``{"words": int, ...}``) or a flat
            ``{short_name: int}`` mapping.
        expected: Reference counts. Defaults to :data:`EXPECTED_WORD_COUNTS`.
        tolerance: Maximum allowed fractional deviation (0.05 = ±5%).

    Returns:
        A pandas DataFrame with columns
        ``[doc, expected, actual, diff, diff_pct, within_tolerance]``,
        sorted by ``doc``. The last row is a ``"TOTAL"`` summary.

    Raises:
        AssertionError: If any single document is outside the tolerance.
            The summary table is included in the assertion message.
    """
    expected = expected or EXPECTED_WORD_COUNTS

    # Normalise actual to a flat {short_name: int} mapping.
    flat_actual: dict[str, int] = {}
    for short_name, value in actual.items():
        if isinstance(value, dict):
            flat_actual[short_name] = int(value["words"])
        else:
            flat_actual[short_name] = int(value)

    rows: list[dict] = []
    for short_name, exp_words in expected.items():
        act_words = flat_actual.get(short_name)
        if act_words is None:
            raise KeyError(
                f"verify_word_counts: missing actual count for '{short_name}'"
            )
        diff = act_words - exp_words
        diff_pct = diff / exp_words if exp_words else float("inf")
        rows.append(
            {
                "doc": short_name,
                "expected": exp_words,
                "actual": act_words,
                "diff": diff,
                "diff_pct": diff_pct,
                "within_tolerance": abs(diff_pct) <= tolerance,
            }
        )

    df = pd.DataFrame(rows).sort_values("doc").reset_index(drop=True)

    # Totals row.
    total_expected = int(df["expected"].sum())
    total_actual = int(df["actual"].sum())
    total_diff = total_actual - total_expected
    total_diff_pct = total_diff / total_expected if total_expected else float("inf")
    total_row = pd.DataFrame(
        [
            {
                "doc": "TOTAL",
                "expected": total_expected,
                "actual": total_actual,
                "diff": total_diff,
                "diff_pct": total_diff_pct,
                "within_tolerance": abs(total_diff_pct) <= tolerance,
            }
        ]
    )
    df = pd.concat([df, total_row], ignore_index=True)

    # Assert per-document tolerance (excluding the TOTAL row, which is
    # informational — a per-doc failure will already have triggered).
    failed = df[(df["doc"] != "TOTAL") & (~df["within_tolerance"])]
    if not failed.empty:
        raise AssertionError(
            "verify_word_counts: one or more documents outside ±"
            f"{tolerance:.0%} tolerance.\n{df.to_string(index=False)}"
        )
    return df


# --------------------------------------------------------------------------- #
# Convenience: format a comparison table for notebook display                  #
# --------------------------------------------------------------------------- #


def format_comparison_table(df: pd.DataFrame) -> str:
    """Render the ``verify_word_counts`` DataFrame as an aligned plaintext table.

    Useful for printing in notebooks where the default pandas repr is noisy.

    Args:
        df: DataFrame returned by :func:`verify_word_counts`.

    Returns:
        Multi-line string ready to ``print(...)``.
    """
    formatted = df.copy()
    formatted["expected"] = formatted["expected"].map(lambda x: f"{x:>9,d}")
    formatted["actual"] = formatted["actual"].map(lambda x: f"{x:>9,d}")
    formatted["diff"] = formatted["diff"].map(lambda x: f"{x:>+7,d}")
    formatted["diff_pct"] = formatted["diff_pct"].map(lambda x: f"{x:>+7.2%}")
    formatted["within_tolerance"] = formatted["within_tolerance"].map(
        lambda x: "OK" if x else "FAIL"
    )
    return formatted.to_string(index=False)


__all__ = [
    "PDF_TO_SHORT_NAME",
    "EXPECTED_WORD_COUNTS",
    "clean_extracted_text",
    "extract_pdf_text",
    "extract_corpus",
    "verify_word_counts",
    "format_comparison_table",
]
