"""Ollama wrapper for the SD-A-01-RAG corpus (Phase B.6).

See MASTER_PLAN.md §8 B.6.

Determinism (§13.5): every call sets ``temperature=0`` and ``seed=42`` by
default. These are passed through to Ollama's options so the LLM produces
identical answers across repeated runs of the same configuration. This is
load-bearing for the three-repetition policy in §10 D.4 — variance in
latency reflects machine noise, not generation noise.

Default LLM: ``llama3.1:8b-instruct-q4_K_M``. The Phi-3.5 backup is selected
by passing ``model_name="phi3.5:3.8b-mini-instruct-q4_K_M"``.

Prompt template is the §8 B.6 specification verbatim. Retrieved chunks are
inlined with light provenance markers (``source`` and ``id``) so the model
sees which doc each chunk came from — useful for the multi-doc question
(Q13) and harmless for single-doc questions.

Public API:
    - :func:`build_prompt` — format the §8 B.6 template.
    - :func:`generate_from_prompt` — Ollama call given a pre-built prompt.
    - :func:`generate` — convenience wrapper that builds the prompt then
      calls :func:`generate_from_prompt`.

Phase C note (refactor): :func:`generate` was previously a single function
that built the prompt and ran the LLM together. It is now a thin wrapper over
``build_prompt`` + ``generate_from_prompt``. This split lets
``src/pipeline.py`` capture the prompt text (for token counting) without
re-building it. The prompt produced by ``generate(query, retrieved, ...)``
remains byte-identical to the pre-refactor behaviour because ``build_prompt``
is unchanged and called the same way.
"""

from __future__ import annotations

from typing import Sequence, Union

import ollama

from src.chunking import Chunk
from src.metrics import time_it
from src.retrieval import RetrievalResult

# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #

DEFAULT_LLM: str = "llama3.1:8b-instruct-q4_K_M"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_SEED: int = 42


# --------------------------------------------------------------------------- #
# Prompt template (§8 B.6, verbatim)                                          #
# --------------------------------------------------------------------------- #

PROMPT_TEMPLATE: str = (
    "Answer the question using ONLY the context provided. If the answer is "
    "not in the context, say \"I don't have enough information to answer "
    "that.\"\n"
    "\n"
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {query}\n"
    "\n"
    "Answer:"
)


# --------------------------------------------------------------------------- #
# Prompt construction                                                         #
# --------------------------------------------------------------------------- #


RetrievedItem = Union[RetrievalResult, Chunk]


def build_prompt(
    query: str,
    retrieved: Sequence[RetrievedItem],
) -> str:
    """Format the §8 B.6 prompt template with the retrieved chunks inlined.

    Each chunk is prefixed with a small provenance line of the form
    ``[chunk N] (source: <source_doc>, id: <chunk_id>)`` followed by the
    chunk text. Chunks are separated by a blank line.

    Args:
        query: The user question.
        retrieved: Either ``RetrievalResult`` instances (from
            :func:`src.retrieval.retrieve`) or bare ``Chunk`` instances.

    Returns:
        The fully formatted prompt string ready for Ollama.
    """
    pieces: list[str] = []
    for i, item in enumerate(retrieved, start=1):
        chunk = item.chunk if isinstance(item, RetrievalResult) else item
        pieces.append(
            f"[chunk {i}] (source: {chunk.source_doc}, id: {chunk.chunk_id})\n"
            f"{chunk.text}"
        )
    context = "\n\n".join(pieces)
    return PROMPT_TEMPLATE.format(context=context, query=query)


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #


@time_it("generate")
def generate_from_prompt(
    prompt: str,
    model_name: str = DEFAULT_LLM,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
) -> str:
    """Run a single Ollama generation against a pre-built prompt.

    This is the LLM-call layer. Callers that need the prompt text for
    measurement (e.g. ``src/pipeline.py`` for prompt token counting) build
    the prompt themselves via :func:`build_prompt` and pass it in here, so
    the prompt is built exactly once per query.

    Args:
        prompt: The fully formatted prompt string (e.g. from
            :func:`build_prompt`).
        model_name: Ollama model tag (e.g. ``llama3.1:8b-instruct-q4_K_M``).
        temperature: LLM temperature. Default 0.0 for determinism.
        seed: LLM seed. Default 42 — see §13.5.

    Returns:
        The generated answer text (with surrounding whitespace stripped).
    """
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"temperature": temperature, "seed": seed},
    )
    # Newer ollama-python returns a Pydantic model; older versions return a
    # plain dict. Normalize via model_dump if available.
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    text = response["response"]
    return text.strip()


def generate(
    query: str,
    retrieved: Sequence[RetrievedItem],
    model_name: str = DEFAULT_LLM,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
) -> str:
    """Build the prompt then run a single Ollama generation.

    Convenience wrapper preserved for back-compat with Phase B callers. New
    code that wants to measure prompt tokens or re-use the prompt string
    should call :func:`build_prompt` + :func:`generate_from_prompt` directly.

    The prompt produced here is byte-identical to the pre-refactor
    ``generate`` because :func:`build_prompt` is unchanged and called the
    same way.

    Args:
        query: The user question.
        retrieved: Retrieved chunks (RetrievalResult or Chunk instances).
        model_name: Ollama model tag (e.g. ``llama3.1:8b-instruct-q4_K_M``).
        temperature: LLM temperature. Default 0.0 for determinism.
        seed: LLM seed. Default 42 — see §13.5.

    Returns:
        The generated answer text (with surrounding whitespace stripped).
    """
    prompt = build_prompt(query, retrieved)
    return generate_from_prompt(
        prompt,
        model_name=model_name,
        temperature=temperature,
        seed=seed,
    )


__all__ = [
    "DEFAULT_LLM",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_SEED",
    "PROMPT_TEMPLATE",
    "build_prompt",
    "generate_from_prompt",
    "generate",
]
