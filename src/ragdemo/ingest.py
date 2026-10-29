"""Ingest pipeline: load -> chunk -> skip known hashes -> embed new chunks -> idempotent upsert."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import psycopg

from ragdemo import store
from ragdemo.chunking import chunk_records
from ragdemo.embedder import Embedder, EmbedResult
from ragdemo.loaders import discover, load_file, source_name
from ragdemo.models import Chunk


@dataclass
class IngestReport:
    files: int = 0
    chunks_total: int = 0
    already_present: int = 0
    embedded: int = 0
    inserted: int = 0
    embed_tokens: int | None = None
    warnings: list[str] = field(default_factory=list)


def ingest_path(
    conn: psycopg.Connection,
    embedder: Embedder,
    path: Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    batch_size: int = 100,
    log: Callable[[str], None] = lambda _msg: None,
) -> IngestReport:
    report = IngestReport()
    files = discover(path)
    if not files:
        report.warnings.append(f"no supported files (.pdf .md .txt .csv) found under {path}")
        return report

    for file in files:
        source = source_name(file, path)
        records, empty_pages = load_file(file, source)
        if empty_pages:
            report.warnings.append(f"{source}: {empty_pages} page(s) had no extractable text (scanned?), skipped")
        chunks = chunk_records(records, chunk_size, chunk_overlap)
        # A file can repeat identical rows; keep the first of each hash.
        unique: dict[str, Chunk] = {c.content_hash: c for c in reversed(chunks)}
        chunks = [c for c in chunks if unique[c.content_hash] is c]

        known = store.existing_hashes(conn, [c.content_hash for c in chunks], embedder.model)
        new = [c for c in chunks if c.content_hash not in known]
        report.files += 1
        report.chunks_total += len(chunks)
        report.already_present += len(chunks) - len(new)
        log(f"{source}: {len(chunks)} chunks, {len(chunks) - len(new)} already stored, {len(new)} new")

        # Embed and commit batch by batch so a failure part-way leaves a resumable state.
        for i in range(0, len(new), batch_size):
            batch = new[i : i + batch_size]
            result = embedder.embed_documents([c.content for c in batch])
            keep = [j for j, v in enumerate(result.vectors) if any(v)]
            if len(keep) < len(batch):
                report.warnings.append(
                    f"{source}: {len(batch) - len(keep)} chunk(s) had a zero embedding (no usable text), skipped"
                )
                batch = [batch[j] for j in keep]
                result = EmbedResult(vectors=[result.vectors[j] for j in keep], tokens=result.tokens)
            report.embedded += len(batch)
            if result.tokens is not None:
                report.embed_tokens = (report.embed_tokens or 0) + result.tokens
            report.inserted += store.upsert_chunks(conn, batch, result.vectors, embedder.model)
    return report
