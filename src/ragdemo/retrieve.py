"""Hybrid retrieval: vector top-k + full-text top-k, fused with reciprocal rank fusion."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Sequence

import psycopg

from ragdemo import store
from ragdemo.embedder import Embedder
from ragdemo.models import RetrievedChunk


def rrf_fuse(
    vector_hits: Sequence[RetrievedChunk],
    fts_hits: Sequence[RetrievedChunk],
    k: int = 60,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank). Ranks are 1-based.
    Merges the per-list details (similarity, keyword score) onto one chunk per id."""
    merged: dict[int, RetrievedChunk] = {}
    scores: dict[int, float] = {}
    for hits in (vector_hits, fts_hits):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            prev = merged.get(hit.id)
            if prev is None:
                merged[hit.id] = hit
            else:
                merged[hit.id] = replace(
                    prev,
                    vector_similarity=prev.vector_similarity if prev.vector_similarity is not None else hit.vector_similarity,
                    vector_rank=prev.vector_rank if prev.vector_rank is not None else hit.vector_rank,
                    fts_score=prev.fts_score if prev.fts_score is not None else hit.fts_score,
                    fts_rank=prev.fts_rank if prev.fts_rank is not None else hit.fts_rank,
                )
    # Highest fused score first; ties go to the better vector rank, then the lower id.
    order = sorted(
        merged,
        key=lambda i: (-scores[i], merged[i].vector_rank or 10**9, i),
    )
    fused = [replace(merged[i], rrf_score=scores[i]) for i in order]
    return fused[:limit] if limit else fused


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]  # fused top-k
    top_similarity: float | None  # best cosine similarity among vector candidates
    vector_hits: int
    fts_hits: int
    embed_ms: float
    search_ms: float
    embed_tokens: int | None
    vector_ranked: list[RetrievedChunk] = field(default_factory=list)  # full candidate lists, for evals
    fts_ranked: list[RetrievedChunk] = field(default_factory=list)


def retrieve(
    conn: psycopg.Connection,
    embedder: Embedder,
    question: str,
    *,
    top_k: int = 5,
    candidates: int = 20,
    rrf_k: int = 60,
    source: str | None = None,
) -> RetrievalResult:
    t0 = time.perf_counter()
    embedded = embedder.embed_query(question)
    t1 = time.perf_counter()
    query_vec = embedded.vectors[0]
    # A zero vector has no direction: cosine distance to it is NaN. Skip the vector side and let
    # keyword search (and the fail-closed gate, which sees no similarity) decide.
    has_direction = any(query_vec)
    vector_hits = store.vector_search(conn, query_vec, embedder.model, candidates, source) if has_direction else []
    fts_hits = store.fts_search(conn, question, embedder.model, candidates, source)
    fused = rrf_fuse(vector_hits, fts_hits, k=rrf_k, limit=top_k)
    t2 = time.perf_counter()
    return RetrievalResult(
        chunks=fused,
        top_similarity=vector_hits[0].vector_similarity if vector_hits else None,
        vector_hits=len(vector_hits),
        fts_hits=len(fts_hits),
        embed_ms=(t1 - t0) * 1000,
        search_ms=(t2 - t1) * 1000,
        embed_tokens=embedded.tokens,
        vector_ranked=vector_hits,
        fts_ranked=fts_hits,
    )
