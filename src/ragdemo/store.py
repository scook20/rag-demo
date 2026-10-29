"""Postgres + pgvector access: migrations, idempotent upsert, vector and keyword search."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Sequence

import psycopg
from psycopg.rows import dict_row

from ragdemo.models import Chunk, RetrievedChunk


def connect(url: str) -> psycopg.Connection:
    return psycopg.connect(url, row_factory=dict_row)


def vector_literal(vec: Sequence[float]) -> str:
    """pgvector text format, passed as a parameter and cast with ::vector."""
    return "[" + ",".join(f"{x:.8g}" for x in vec) + "]"


# ---- migrations -------------------------------------------------------------------------------


def migrate(conn: psycopg.Connection, migrations_dir: Path) -> list[str]:
    """Apply any *.sql files not yet recorded in schema_migrations, in filename order."""
    applied_now: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(727274)")  # serialize concurrent runners
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        cur.execute("SELECT filename FROM schema_migrations")
        done = {row["filename"] for row in cur.fetchall()}
        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in done:
                continue
            cur.execute(path.read_text(encoding="utf-8"))
            cur.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            applied_now.append(path.name)
    conn.commit()
    return applied_now


def schema_ready(conn: psycopg.Connection) -> bool:
    """True once the migrations have created the chunks table."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.chunks') IS NOT NULL AS ready")
        ready = bool(cur.fetchone()["ready"])
    conn.rollback()
    return ready


def check_dimension(conn: psycopg.Connection, expected: int) -> None:
    """The vector column dimension is fixed by the migration; fail early if EMBEDDING_DIM disagrees."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT atttypmod AS dim FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )
        row = cur.fetchone()
    conn.rollback()
    if row and row["dim"] != expected:
        raise RuntimeError(f"chunks.embedding is vector({row['dim']}) but EMBEDDING_DIM={expected}")


# ---- write ------------------------------------------------------------------------------------


def existing_hashes(conn: psycopg.Connection, hashes: Sequence[str], model: str) -> set[str]:
    if not hashes:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content_hash FROM chunks WHERE embedding_model = %s AND content_hash = ANY(%s)",
            (model, list(hashes)),
        )
        found = {row["content_hash"] for row in cur.fetchall()}
    conn.rollback()
    return found


def upsert_chunks(
    conn: psycopg.Connection, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], model: str
) -> int:
    """Insert chunks keyed by (content_hash, embedding_model). Existing keys are left untouched.
    Returns the number of rows actually inserted."""
    inserted = 0
    with conn.cursor() as cur:
        for chunk, vec in zip(chunks, vectors, strict=True):
            cur.execute(
                "INSERT INTO chunks (content_hash, embedding_model, source, page, chunk_index, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector) "
                "ON CONFLICT (content_hash, embedding_model) DO NOTHING RETURNING id",
                (
                    chunk.content_hash,
                    model,
                    chunk.source,
                    chunk.page,
                    chunk.chunk_index,
                    chunk.content,
                    json.dumps(chunk.metadata),
                    vector_literal(vec),
                ),
            )
            if cur.fetchone() is not None:
                inserted += 1
    conn.commit()
    return inserted


def count_chunks(conn: psycopg.Connection, model: str | None = None) -> int:
    with conn.cursor() as cur:
        if model:
            cur.execute("SELECT count(*) AS n FROM chunks WHERE embedding_model = %s", (model,))
        else:
            cur.execute("SELECT count(*) AS n FROM chunks")
        n = cur.fetchone()["n"]
    conn.rollback()
    return n


# ---- search -----------------------------------------------------------------------------------

_COLUMNS = "id, source, page, chunk_index, content, metadata"


def _row_to_chunk(row: dict, **extra) -> RetrievedChunk:
    return RetrievedChunk(
        id=row["id"],
        source=row["source"],
        page=row["page"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        metadata=row["metadata"] or {},
        **extra,
    )


def vector_search(
    conn: psycopg.Connection,
    query_vec: Sequence[float],
    model: str,
    k: int,
    source: str | None = None,
) -> list[RetrievedChunk]:
    """Top-k by cosine distance (<=>) using the HNSW index. similarity = 1 - distance."""
    params: dict = {"q": vector_literal(query_vec), "model": model, "k": k}
    where = "embedding_model = %(model)s"
    if source:
        where += " AND source = %(source)s"
        params["source"] = source
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL hnsw.ef_search = {max(40, int(k))}")
        if source:
            # Filters run after the index scan; iterative scans (pgvector 0.8.0+) keep scanning
            # until enough rows pass the filter.
            cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        cur.execute(
            f"SELECT {_COLUMNS}, 1 - (embedding <=> %(q)s::vector) AS similarity "
            f"FROM chunks WHERE {where} ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
            params,
        )
        rows = cur.fetchall()
    conn.rollback()
    hits = [(r, float(r["similarity"])) for r in rows if not math.isnan(float(r["similarity"]))]  # NaN: zero vector
    hits.sort(key=lambda pair: pair[1], reverse=True)  # relaxed_order can return slightly out of order
    return [_row_to_chunk(r, vector_similarity=sim, vector_rank=i) for i, (r, sim) in enumerate(hits, start=1)]


def keyword_query(question: str) -> str:
    """Turn a natural-language question into a websearch_to_tsquery string that ORs the terms.
    websearch_to_tsquery ANDs bare words, which returns nothing for most questions."""
    words: list[str] = []
    for word in re.findall(r"[A-Za-z0-9]+", question):
        if word.lower() != "or" and word.lower() not in (w.lower() for w in words):
            words.append(word)
    return " or ".join(words)


def fts_search(
    conn: psycopg.Connection, question: str, model: str, k: int, source: str | None = None
) -> list[RetrievedChunk]:
    """Top-k by full-text rank (ts_rank_cd, length-normalised) against the GIN-indexed tsvector."""
    query = keyword_query(question)
    if not query:
        return []
    params: dict = {"q": query, "model": model, "k": k}
    where = "embedding_model = %(model)s AND content_tsv @@ q.query"
    if source:
        where += " AND source = %(source)s"
        params["source"] = source
    with conn.cursor() as cur:
        cur.execute(
            f"WITH q AS (SELECT websearch_to_tsquery('english'::regconfig, %(q)s) AS query) "
            f"SELECT {_COLUMNS}, ts_rank_cd(content_tsv, q.query, 32) AS score "
            f"FROM chunks, q WHERE {where} ORDER BY score DESC, id LIMIT %(k)s",
            params,
        )
        rows = cur.fetchall()
    conn.rollback()
    return [_row_to_chunk(r, fts_score=float(r["score"]), fts_rank=i) for i, r in enumerate(rows, start=1)]
