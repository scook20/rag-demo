"""Split text into overlapping chunks that prefer paragraph and sentence boundaries."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from ragdemo.models import Chunk, Record

_SEPARATORS = ("\n\n", "\n", ". ", " ")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Greedy window splitter. Each chunk is at most `size` characters and starts
    roughly `overlap` characters before the previous chunk ended."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    text = normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in _SEPARATORS:
                cut = window.rfind(sep)
                if cut >= size // 2:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # Do not begin mid-word: move forward to the next word boundary.
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        start = next_start
    return chunks


def content_hash(source: str, page: int | None, row: int | None, content: str) -> str:
    key = "\x1f".join([source, "" if page is None else str(page), "" if row is None else str(row), content])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def chunk_records(records: Iterable[Record], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Chunk records. chunk_index counts per (source, page) group; structured rows keep their own text."""
    chunks: list[Chunk] = []
    counters: dict[tuple[str, int | None], int] = {}
    for rec in records:
        pieces = chunk_text(rec.text, size, overlap) if rec.chunkable else [normalize(rec.text)]
        for piece in pieces:
            if not piece:
                continue
            key = (rec.source, rec.page)
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            meta = dict(rec.metadata)
            if rec.chunkable and rec.page is None and len(pieces) > 1:
                meta["section"] = idx + 1
            chunks.append(
                Chunk(
                    source=rec.source,
                    page=rec.page,
                    chunk_index=idx,
                    content=piece,
                    content_hash=content_hash(rec.source, rec.page, rec.metadata.get("row"), piece),
                    metadata=meta,
                )
            )
    return chunks
