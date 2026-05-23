# SD-A-01-RAG — The Footprint of Retrieval-Augmented Generation Components

A controlled, single-machine study of how chunk size, embedding model, and retrieval depth affect the **cost** and **quality** of a small Retrieval-Augmented Generation (RAG) pipeline — with the corpus itself drawn from recent literature on the environmental cost of AI.

> An AI system measuring the footprint of its own components, while answering questions about the footprint of AI.

**Course:** Sustainable Data — The Environmental Cost of the Digital World
**Institution:** University of Europe for Applied Sciences (UE Germany), Summer Semester 2026
**Instructor:** Prof. Dr. Iftikhar Ahmed
**Team SD-A-01:**

- Tolga Özcan — Team Lead; coordination of writing and milestone delivery
- Saheed Yakubu — Corpus selection and evaluation set construction
- İsmail Demircan — Pipeline implementation and measurement layer
- Anandhu Rajappan Krishnan — Experimental execution and plotting
- Eze Nnaemeka Uzoma — Manual grading and paper drafting

---

## What this project does

We build a fixed RAG pipeline over five sustainability/AI documents (BLOOM, IEA *Energy and AI*, Google 2025 Environmental Report, EPRI *Powering Intelligence*, Greenpeace *Chipping Point*) and a fixed evaluation set of 13 questions stratified by document and question type. We then run three controlled mini-experiments — varying (1) chunk size and overlap, (2) embedding model, and (3) top-k retrieval depth — and measure both **indexing-time** and **query-time** costs (latency, memory, storage, tokens, energy via CodeCarbon) alongside quality (recall@k stratified by question type plus a 1–5 manual grading rubric with atomic-fact partial credit).

All measurements come from a single hardware configuration (MacBook Air M4, 16 GB RAM, macOS) to maximise internal validity. Cross-configuration trade-offs are the point; cross-machine absolute numbers are not.

## Stack

- Python 3.11 (use `venv`, `conda`, or your preferred environment manager)
- LangChain · FAISS-CPU · sentence-transformers
- Ollama serving **Llama 3.1 8B Instruct (Q4_K_M)** as primary LLM, **Phi-3.5 Mini (Q4_K_M)** as backup
- CodeCarbon for energy and emissions estimation
- pandas, matplotlib, seaborn for analysis and figures
- scikit-learn for Cohen's κ

## Install

Tested on macOS (Apple Silicon), but the stack is cross-platform — every dependency is available for macOS, Linux, and Windows.

```bash
# 1. Create and activate a Python 3.11 environment.
#    Option A — venv (Python's built-in, cross-platform):
python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell or cmd)

#    Option B — conda (if you already use it):
# conda create -n sd-a-01-rag python=3.11
# conda activate sd-a-01-rag

# 2. Install Python dependencies.
pip install -r requirements.txt --upgrade

# 3. Install Ollama (https://ollama.com — available for macOS, Linux, Windows)
#    and pull the local models.
ollama pull llama3.1:8b-instruct-q4_K_M       # ~4.9 GB
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M  # ~2.2 GB

# 4. Verify both models are listed.
ollama list

# 5. Launch JupyterLab.
jupyter lab
```

Open `notebooks/00_setup_check.ipynb` first to verify the environment, then proceed through the numbered notebooks in order.

## Repository layout

```
dataset/         # 5 corpus PDFs, evaluation questions, atomic-fact rubric
src/             # reusable Python modules (chunking, embedding, indexing, retrieval, generation, metrics, grading, recall)
notebooks/       # one notebook per phase, numbered for run order (00 setup → 08 analysis)
results/         # indexing_log.csv, query_log.csv, grading_log.csv, kappa_summary.csv
plots/           # the four headline figures (3 trade-off plots + 1 stratified-recall plot)
outputs/         # course deliverables (proposal, progress review, poster, final presentation, paper)
requirements.txt
AI_USE.md        # AI Use Declaration
CITATION.cff     # cite-this-repo metadata
LICENSE          # code license
```

Two derived directories are regenerated locally and not committed: `indices/` (FAISS bundles per configuration) and `corpus_text/` (plain-text extractions from the PDFs).

## Reproducibility

Three outputs of this project reproduce to three different degrees. We document this honestly because conflating them would mislead anyone trying to verify our results.

| Output | Reproduces on another machine? | Why |
|--------|--------------------------------|-----|
| **Recall@k, Cohen's κ, the 4 plots** | **Yes, identically.** | Derived from the committed CSVs in `results/`. Re-running `notebooks/08_analysis_and_plots.ipynb` against the same `query_log.csv` + `grading_log.csv` produces pixel-identical plots and κ = 0.9425. |
| **Answer text from the LLM** | **Yes on similar hardware; possibly small divergence across architectures.** | All LLM calls use `temperature=0, seed=42` and chunking/embedding are deterministic. On another M-series Mac the answers should be byte-identical. On Linux/x86 or NVIDIA GPU, floating-point reductions can run in a different order and occasionally tip a near-tie token. This is a known LLM-determinism limitation, not a pipeline bug. |
| **Latency, energy (Wh), peak RAM** | **No — by design.** | We follow a single-machine measurement policy to maximise internal validity for the cross-configuration trade-offs. These numbers are valid within-machine, not across. |

### Pinning for byte-identical model weights

Each Ollama model tag resolves to a content-addressed ID that is identical for everyone who pulls it. To verify after `ollama pull`:

```bash
ollama list
# The ID column should match the values recorded below.
```

**Ollama version used:** `0.23.2`
**Llama 3.1 model ID:** `46e0c10c039e` (tag `llama3.1:8b-instruct-q4_K_M`, 4.9 GB)
**Phi-3.5 model ID:** `570961596984` (tag `phi3.5:3.8b-mini-instruct-q4_K_M`, 2.4 GB)

If your local digest matches, the weights are byte-identical to ours.

## How to read the results

- `results/indexing_log.csv` — one row per indexing run; cost paid **once** per configuration (chunking + embedding + FAISS build).
- `results/query_log.csv` — one row per (configuration, question) pair; cost paid **per question** (query embedding + retrieval + generation).
- `results/grading_log.csv` — three graders × 13 questions × measured configurations; partial-credit grade plus the count of atomic facts present.
- `results/kappa_summary.csv` — inter-rater agreement (quadratic-weighted Cohen's κ = 0.9425, "almost perfect agreement" per Landis-Koch).
- `plots/exp1_chunk_overlap_tradeoff.png`, `exp2_embedding_model_tradeoff.png`, `exp3_topk_tradeoff.png` — Pareto-style cost–quality scatter, one per research question.
- `plots/recall_by_question_type.png` — recall@k broken down by factoid / numerical / synthesis / multi-doc.

The headline finding is the Pareto-winner configuration `aadb0cb9` at top-k = 3 (see `outputs/paper/` for the full discussion when published).

## Conventions used in the code

- Random seed: `42` everywhere. LLM `temperature=0`.
- Indexing-time cost and query-time cost are logged in **separate** CSVs — never mixed.
- All measurements come from a single machine; we do not aggregate across machines.
- Code, comments, file names, and writing outputs are English-only.
- Each configuration is identified by a short hash (first 8 chars of SHA1 over `(chunk_size, overlap_pct, embedding_model)`) — used as the folder name in `indices/`.

## AI Use Declaration

This project uses AI assistants as tooling under human direction. The full declaration — covering corpus selection, evaluation question drafting, code generation, plotting, and writing — is in [`AI_USE.md`](./AI_USE.md) and is reproduced in the final research paper.

## License

Code in `src/`, `notebooks/`, and the supporting scripts is released under the MIT License (see [`LICENSE`](./LICENSE)). Authored documents in `dataset/` (evaluation questions, atomic-fact rubric, corpus statistics, source notes) are released under CC-BY-4.0; the corpus PDFs themselves retain their original publishers' terms — see `dataset/corpus_sources.md` for per-document provenance.

## Citation

If this work is useful to you, please cite via [`CITATION.cff`](./CITATION.cff). The research paper is forthcoming (Milestone 5, due 7 July 2026); citation metadata will be updated when it is finalised.

---

_This repository is a course deliverable for **Sustainable Data — The Environmental Cost of the Digital World** at UE Germany. The internal project plan and progress log live outside this public repository and are maintained by the team for execution purposes only._
