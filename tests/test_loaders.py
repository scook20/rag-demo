from pathlib import Path

from ragdemo.loaders import discover, load_csv, load_file, load_pdf, source_name


def test_pdf_keeps_page_numbers(corpus_dir):
    records, empty = load_pdf(corpus_dir / "kestrel_reserves_memo.pdf", "memo.pdf")
    assert empty == 0
    assert [r.page for r in records] == [1, 2]
    assert "2P" in records[0].text and "waterflood" in records[1].text


def test_csv_rows_become_text_and_jsonb_metadata(corpus_dir):
    records = load_csv(corpus_dir / "kestrel_production_2026.csv", "prod.csv")
    assert len(records) == 9
    first = records[0]
    assert first.text.startswith("month: 2026-01; well: Kestrel-2; oil_bbl: 9100")
    assert first.metadata["row"] == 1
    assert first.metadata["columns"]["well"] == "Kestrel-2"
    assert first.chunkable is False


def test_markdown_is_loaded_whole(corpus_dir):
    records, _ = load_file(corpus_dir / "kestrel_operations_notes.md", "notes.md")
    assert len(records) == 1 and "ESP" in records[0].text


def test_discover_sorted_and_skips_hidden(tmp_path):
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "ignore.bin").write_text("x")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c.md").write_text("c")
    names = [p.name for p in discover(tmp_path)]
    assert names == ["a.txt", "b.md"]


def test_source_name_relative_to_root(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    f = sub / "x.md"
    f.write_text("x")
    assert source_name(f, tmp_path) == "sub/x.md"
    assert source_name(f, f) == "x.md"


def test_empty_csv_rows_skipped(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("a,b\n,\n1,2\n")
    assert len(load_csv(f, "e.csv")) == 1
