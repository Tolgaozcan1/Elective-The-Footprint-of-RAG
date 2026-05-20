# Corpus Statistics

This corpus supports a group project for the course **Sustainable Data – The
Environmental Cost of the Digital World** (Summer Semester 2026, University of
Europe for Applied Sciences). The project investigates the footprint of
Retrieval-Augmented Generation (RAG) components — chunk size, overlap, embedding
model, and retrieval depth — and their effects on storage, compute, and answer
quality. The corpus comprises five documents spanning academic research,
international organization reports, tech-company sustainability disclosures,
independent research, and NGO investigations — intentionally heterogeneous in
length and detail.

**Note on evaluation design:** The corpus is intentionally heterogeneous in length (ranging
from 15 to 304 pages). Rather than truncating longer documents to equalize token counts,
evaluation is balanced through stratified question allocation: shorter documents receive
fewer questions and longer documents receive more, ensuring each document is adequately
represented without artificially distorting the corpus.

---

## Per-Document Statistics

| # | Short name | Pages | Words | Characters | File size (MB) |
|---|-----------|------:|------:|-----------:|---------------:|
| 1 | BLOOM | 15 | 6,939 | 44,622 | 0.24 |
| 2 | IEA | 304 | 115,129 | 789,841 | 7.83 |
| 3 | Google | 120 | 56,660 | 386,368 | 18.55 |
| 4 | EPRI | 35 | 14,264 | 101,269 | 3.25 |
| 5 | Greenpeace | 27 | 7,025 | 51,907 | 2.44 |
| **Total** | | **501** | **200,017** | **1,374,007** | **32.31** |

---

## Notes

- Word counts computed via `pypdf` text extraction: `len(extracted_text.split())`.
- Character counts include whitespace as extracted by `pypdf`.
- File sizes are as-stored on disk (compressed PDF); text content may differ from raw-text equivalents.
- The IEA report (304 pages) dominates the corpus by word count; BLOOM (15 pages) is the smallest document.
