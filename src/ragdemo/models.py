"""Plain data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Record:
    """One unit of text from a loader, before chunking."""

    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunkable: bool = True  # structured rows (CSV) stay whole


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int | None
    chunk_index: int
    content: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    source: str
    page: int | None
    chunk_index: int
    content: str
    metadata: dict[str, Any]
    vector_similarity: float | None = None  # cosine similarity, None if not a vector hit
    vector_rank: int | None = None
    fts_score: float | None = None  # ts_rank_cd, None if not a keyword hit
    fts_rank: int | None = None
    rrf_score: float = 0.0

    @property
    def label(self) -> str:
        return citation_label(self.source, self.page, self.metadata)


def citation_label(source: str, page: int | None, metadata: dict[str, Any]) -> str:
    """The inline citation text, for example `report.pdf p.4` or `production.csv row 3`."""
    if page is not None:
        return f"{source} p.{page}"
    row = metadata.get("row")
    if row is not None:
        return f"{source} row {row}"
    section = metadata.get("section")
    if section is not None:
        return f"{source} section {section}"
    return source
