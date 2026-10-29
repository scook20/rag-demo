"""Ask: retrieve, fail closed if nothing relevant, otherwise generate a cited answer and log metrics."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg

from ragdemo.config import Settings
from ragdemo.embedder import Embedder
from ragdemo.generator import NOT_IN_CORPUS, Generator, is_refusal
from ragdemo.guard import unsupported_terms
from ragdemo.metrics import append_jsonl, estimate_cost, load_pricing
from ragdemo.models import RetrievedChunk
from ragdemo.retrieve import RetrievalResult, retrieve

_CITATION = re.compile(r"\[([^\[\]]+?)\]")


@dataclass(frozen=True)
class Answer:
    text: str
    answered: bool  # False when the corpus does not cover the question
    chunks: list[RetrievedChunk]
    metrics: dict[str, Any]
    retrieval: RetrievalResult | None = None


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def ask(
    conn: psycopg.Connection,
    embedder: Embedder,
    generator: Generator,
    settings: Settings,
    question: str,
    *,
    source: str | None = None,
    top_k: int | None = None,
    log: bool = True,
) -> Answer:
    t_start = time.perf_counter()
    result: RetrievalResult = retrieve(
        conn,
        embedder,
        question,
        top_k=top_k or settings.top_k,
        candidates=settings.candidates,
        rrf_k=settings.rrf_k,
        source=source,
    )

    # Three layers, cheapest first. 1) best cosine similarity must clear the threshold, 2) every specific
    # term in the question (name, number, month) must appear in the retrieved excerpts, 3) the model
    # itself must not reply with the refusal sentinel. Layers 1 and 2 skip the model call entirely.
    similarity_ok = result.top_similarity is not None and result.top_similarity >= settings.min_similarity
    missing = unsupported_terms(question, result.chunks) if similarity_ok and settings.term_guard else []
    covered = similarity_ok and not missing
    refusal_reason: str | None = None
    generation_ms = 0.0
    gen = None
    if covered:
        t0 = time.perf_counter()
        gen = generator.generate(question, result.chunks)
        generation_ms = (time.perf_counter() - t0) * 1000
        text = gen.text
        if is_refusal(text):
            text = NOT_IN_CORPUS
            refusal_reason = "model_refusal"
    else:
        text = NOT_IN_CORPUS  # fail closed: never call the model without relevant context
        refusal_reason = "low_similarity" if not similarity_ok else "unsupported_terms"
    answered = refusal_reason is None

    cited = [m.strip() for m in _CITATION.findall(text)]
    valid_labels = {c.label for c in result.chunks}
    pricing = load_pricing(settings.pricing_path)

    embed_tokens = result.embed_tokens
    embed_estimated = False
    if embed_tokens is None:
        embed_tokens = max(1, len(question) // 4)  # rough chars/4 when the API reports no count
        embed_estimated = True
    embed_cost = estimate_cost(pricing, embedder.model, embed_tokens)
    gen_cost = estimate_cost(pricing, generator.model, gen.tokens_in, gen.tokens_out) if gen else 0.0
    total_cost = None if embed_cost is None or gen_cost is None else embed_cost + gen_cost

    tokens_out = gen.tokens_out if gen else 0
    synthetic = getattr(generator, "synthetic", False)  # offline fake: timing would be meaningless
    tokens_per_sec = (
        (tokens_out / (generation_ms / 1000)) if gen and tokens_out and generation_ms > 0 and not synthetic else None
    )

    metrics: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": uuid.uuid4().hex[:12],
        "question": question,
        "source_filter": source,
        "answered": answered,
        "grounded": covered,
        "refusal_reason": refusal_reason,
        "unsupported_terms": missing,
        "embedding_model": embedder.model,
        "generation_model": generator.model,
        "latency_ms": {
            "embed_query": _round(result.embed_ms, 1),
            "retrieval": _round(result.search_ms, 1),
            "generation": _round(generation_ms, 1),
            "total": _round((time.perf_counter() - t_start) * 1000, 1),
        },
        "tokens": {
            "embed_in": embed_tokens,
            "embed_in_estimated": embed_estimated,
            "gen_in": gen.tokens_in if gen else 0,
            "gen_out": tokens_out,
            "gen_thinking": gen.tokens_thinking if gen else 0,
        },
        "tokens_per_sec": _round(tokens_per_sec, 1),
        "est_cost_usd": {
            "embedding": _round(embed_cost, 8),
            "generation": _round(gen_cost, 8),
            "total": _round(total_cost, 8),
            "pricing_verified_on": str(pricing.get("verified_on")),
        },
        "scores": {
            "min_similarity": settings.min_similarity,
            "top_similarity": _round(result.top_similarity),
            "vector": [_round(c.vector_similarity) for c in result.chunks],
            "fts": [_round(c.fts_score) for c in result.chunks],
            "rrf": [_round(c.rrf_score, 6) for c in result.chunks],
        },
        "chunk_ids": [c.id for c in result.chunks],
        "citations": {"cited": cited, "all_in_retrieved": all(c in valid_labels for c in cited) if cited else None},
    }
    if log:
        append_jsonl(settings.log_path, metrics)
    return Answer(text=text, answered=answered, chunks=result.chunks, metrics=metrics, retrieval=result)
