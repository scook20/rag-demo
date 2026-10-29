import pytest

from ragdemo.chunking import chunk_records, chunk_text, content_hash, normalize
from ragdemo.models import Record


def test_short_text_is_one_chunk():
    assert chunk_text("A short note.", size=100, overlap=10) == ["A short note."]


def test_empty_text_has_no_chunks():
    assert chunk_text("  \n\n ", size=100, overlap=10) == []


def test_chunks_respect_size_and_overlap_and_cover_all_words():
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share text (overlap)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()[-3:]) & set(b.split()[:8])
    # nothing lost
    assert set(" ".join(chunks).split()) == set(words)


def test_paragraph_boundary_is_preferred():
    para = "alpha " * 30
    text = para.strip() + "\n\n" + ("beta " * 30).strip()
    chunks = chunk_text(text, size=200, overlap=20)
    assert chunks[0].endswith("alpha") and chunks[1].startswith(("alpha", "beta"))


def test_terminates_on_unbroken_text():
    chunks = chunk_text("x" * 5000, size=300, overlap=100)
    assert len(chunks) >= 5
    assert all(len(c) <= 300 for c in chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_normalize_collapses_whitespace():
    assert normalize("a   b\r\n\r\n\r\n\r\nc") == "a b\n\nc"


def test_chunk_records_keep_page_index_and_hash():
    recs = [
        Record("page one text", "doc.pdf", page=1),
        Record("page two text", "doc.pdf", page=2),
    ]
    chunks = chunk_records(recs, 100, 10)
    assert [(c.page, c.chunk_index) for c in chunks] == [(1, 0), (2, 0)]
    assert len({c.content_hash for c in chunks}) == 2


def test_structured_rows_are_not_split():
    long_row = "col: " + "v" * 3000
    chunks = chunk_records([Record(long_row, "t.csv", metadata={"row": 1}, chunkable=False)], 200, 20)
    assert len(chunks) == 1 and len(chunks[0].content) > 200


def test_hash_is_stable_and_depends_on_locator():
    a = content_hash("a.md", None, None, "same text")
    assert a == content_hash("a.md", None, None, "same text")
    assert a != content_hash("b.md", None, None, "same text")
    assert a != content_hash("a.md", 2, None, "same text")
