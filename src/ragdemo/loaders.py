"""Read PDF, Markdown/text, and CSV files into Records."""

from __future__ import annotations

import csv
from pathlib import Path

from pypdf import PdfReader

from ragdemo.models import Record

SUPPORTED = {".pdf", ".md", ".markdown", ".txt", ".csv"}


def discover(path: Path) -> list[Path]:
    """Return supported files under `path` (or `path` itself), sorted, skipping hidden files."""
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED else []
    files = [
        p
        for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED
        and not any(part.startswith(".") for part in p.relative_to(path).parts)
    ]
    return sorted(files)


def source_name(file: Path, root: Path) -> str:
    base = root.parent if root.is_file() else root
    return file.relative_to(base).as_posix()


def load_pdf(file: Path, source: str) -> tuple[list[Record], int]:
    """One Record per page with text. Returns (records, number of pages without extractable text)."""
    reader = PdfReader(str(file))
    records: list[Record] = []
    empty = 0
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            empty += 1
            continue
        records.append(Record(text=text, source=source, page=number, metadata={"file_type": "pdf"}))
    return records, empty


def load_text(file: Path, source: str) -> list[Record]:
    text = file.read_text(encoding="utf-8", errors="replace")
    return [Record(text=text, source=source, metadata={"file_type": file.suffix.lower().lstrip(".")})]


def load_csv(file: Path, source: str) -> list[Record]:
    """Each row becomes `column: value; ...` text. The raw columns are kept as metadata (JSONB)."""
    records: list[Record] = []
    with file.open(newline="", encoding="utf-8-sig") as handle:
        for number, row in enumerate(csv.DictReader(handle), start=1):
            clean = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            text = "; ".join(f"{k}: {v}" for k, v in clean.items() if v)
            if not text:
                continue
            records.append(
                Record(
                    text=text,
                    source=source,
                    metadata={"file_type": "csv", "row": number, "columns": clean},
                    chunkable=False,
                )
            )
    return records


def load_file(file: Path, source: str) -> tuple[list[Record], int]:
    """Returns (records, pages_without_text)."""
    suffix = file.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file, source)
    if suffix == ".csv":
        return load_csv(file, source), 0
    return load_text(file, source), 0
