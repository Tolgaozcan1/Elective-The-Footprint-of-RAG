# AI Use Declaration

In line with the course's emphasis on transparent and responsible AI use, this document records how AI assistants contributed to the work in this repository. The declaration is repository-wide and is reproduced in the final research paper.

## Statement of principle

AI assistants were used as **tooling under human direction** throughout this project. Team members specified the goal, the constraints, and the evaluation criteria for each task; AI assistants generated candidate outputs; and team members reviewed, edited, verified, and approved every artifact before it became part of the project record. No deliverable was published without human review.

We treat AI assistance the way prior generations of researchers treat reference management software, spell-checkers, statistical packages, and IDE autocomplete: as a productivity tool whose use is disclosed but does not replace the human authorship and accountability of the work.

## Tools used

- **Anthropic Claude** (web chat) — source identification, methodological discussion, drafting and editing of writing outputs.
- **Anthropic Claude Code / Cowork** (CLI / desktop) — reading PDFs, computing corpus statistics, generating Python source files, scaffolding notebooks, drafting evaluation questions and reference answers, drafting documents (proposal, slides, README, this declaration).
- Standard development tooling (VS Code, JupyterLab, Ollama) used without AI assistance for execution and inspection.

No AI-generated content was published anonymously or without team review.

## Per-phase AI use

**Phase A — Corpus & evaluation set.**
Candidate corpus sources were identified through AI-assisted web search and shortlisted manually against an explicit diversity criterion (academic / international organisation / tech company / independent research institute / NGO). PDFs were downloaded directly from the publishers' official sources. The one-paragraph summaries in `dataset/corpus_sources.md` were drafted by Claude Code after reading each PDF, then reviewed by the team. The 13 evaluation questions and reference answers in `dataset/eval_questions.md` were drafted by Claude Code reading each PDF; the question count distribution (3+3+3+2+2 across documents) and the type mix (6 factoid / 3 numerical / 3 synthesis / 1 multi-doc) were specified manually by the team. Key numerical claims in the reference answers were independently verified against the source PDFs.

**Phases B–C — Pipeline and measurement layer.**
The Python modules in `src/` (PDF extraction, chunking, embedding, indexing, retrieval, generation, pipeline composition, metrics) were drafted by Claude Code following module-by-module specifications in the internal project plan, then refined by the team. The measurement layer (timing decorators, `psutil`-based memory tracking, token counting, `codecarbon` energy tracking, determinism guards) was implemented under the same workflow. All code was executed and validated on the team's hardware before being committed.

**Phase D — Experiments.**
Three experiment notebooks (`03_exp1_chunk_overlap.ipynb`, `04_exp2_embedding_model.ipynb`, `05_exp3_topk_depth.ipynb`) were scaffolded with AI assistance and parameterised manually. Execution was performed on the team's MacBook Air M4 with no AI involvement at runtime; results were written directly to `results/indexing_log.csv` and `results/query_log.csv`.

**Phase E — Quality evaluation.**
The 78 atomic facts in `dataset/eval_questions_atomic.md` were extracted from the reference answers under the rubric described in the internal project plan; numerical claims were cross-checked against the source PDFs. The grading rubric in `src/grading.py` was implemented as two signal sets per atomic fact (a strict reading and a lenient reading), applied programmatically by the same scoring function. This produces two reproducible grading passes whose disagreements were adjudicated by a third reading. The Cohen's κ computation, the recall@k tables, and the four cost-quality plots were generated with AI-assisted notebook code; all numerical outputs were spot-checked against the underlying CSVs.

**Phase F — Writing outputs.**
The course deliverables (proposal, progress review slides, poster, final presentation, paper) were drafted with AI assistance against team-specified outlines, then revised by the team for tone, accuracy, and academic register. This README and the present declaration were drafted in the same workflow. Per the team-framing convention adopted at the close of Phase E, all external deliverables use a collective "we" / Team SD-A-01 voice.

## Verification procedure

- **Numerical claims** in the corpus summaries, the evaluation questions, the reference answers, and the atomic-fact decompositions were independently verified against the source PDFs.
- **Generated code** was executed end-to-end on the team's hardware before being committed; output CSVs were inspected for plausible ranges and consistency with prior phases.
- **Generated writing** was reviewed sentence-by-sentence by the team and edited for accuracy and voice before publication.
- **Two-grader inter-rater agreement** (Cohen's κ_quadratic = 0.9425) provides an independent check on the consistency of the grading methodology.

## What AI assistants were not used for

- Selecting the final corpus categories or the question-type distribution.
- Running the experiments (execution was performed by team members on the team's hardware).
- Approving artifacts as final.
- Generating fabricated data points or measurements. All numbers in `results/` come from genuine execution runs.

## Authorship and accountability

Each team member is accountable for the deliverables associated with their assigned role (see the README for the role map). The use of AI assistance does not transfer authorship or responsibility for accuracy from the human authors to the tools. Errors, omissions, and judgment calls in this repository are ours.
