"""Hits the local pgvector container. Skipped cleanly when the database is not reachable."""

import json

from ragdemo import store
from ragdemo.answer import ask
from ragdemo.embedder import FakeEmbedder
from ragdemo.generator import NOT_IN_CORPUS, FakeGenerator
from ragdemo.ingest import ingest_path


def _ingest(db, corpus_dir, settings):
    return ingest_path(db, FakeEmbedder(settings.embedding_dim), corpus_dir, batch_size=4)


def test_migration_creates_hnsw_and_gin_indexes_and_is_idempotent(db, fake_settings):
    assert store.migrate(db, fake_settings.migrations_dir) == []  # already applied by the fixture
    with db.cursor() as cur:
        cur.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks'")
        defs = " ".join(r["indexdef"] for r in cur.fetchall())
    assert "USING hnsw" in defs and "vector_cosine_ops" in defs and "USING gin" in defs
    store.check_dimension(db, 768)


def test_ingest_is_idempotent(db, corpus_dir, fake_settings):
    first = _ingest(db, corpus_dir, fake_settings)
    assert first.files == 7 and first.inserted == first.embedded == first.chunks_total > 0
    n = store.count_chunks(db)
    second = _ingest(db, corpus_dir, fake_settings)
    assert second.inserted == 0 and second.embedded == 0 and second.already_present == first.chunks_total
    assert store.count_chunks(db) == n


def test_ingest_stores_pages_and_csv_metadata(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)
    with db.cursor() as cur:
        cur.execute("SELECT DISTINCT page FROM chunks WHERE source = 'kestrel_reserves_memo.pdf' ORDER BY page")
        assert [r["page"] for r in cur.fetchall()] == [1, 2]
        cur.execute("SELECT metadata FROM chunks WHERE source = 'kestrel_production_2026.csv' AND metadata->>'row' = '6'")
        meta = cur.fetchone()["metadata"]
    assert meta["columns"]["well"] == "Kestrel-4" and meta["columns"]["uptime_pct"] == "52.3"


def test_hybrid_ask_answers_with_citation_and_logs_metrics(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)
    result = ask(db, FakeEmbedder(768), FakeGenerator(), fake_settings, "What are the 2P reserves of Kestrel Field?")
    assert result.answered
    assert "[kestrel_reserves_memo.pdf p.1]" in result.text
    assert result.chunks[0].vector_similarity is not None and result.chunks[0].fts_score is not None
    lines = fake_settings.log_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["answered"] and rec["chunk_ids"] and rec["citations"]["all_in_retrieved"] is True
    assert {"embed_query", "retrieval", "generation", "total"} <= set(rec["latency_ms"])
    assert rec["est_cost_usd"]["pricing_verified_on"] == "2026-10-21"


def test_fails_closed_when_corpus_does_not_cover(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)

    class ExplodingGenerator:
        model = "should-not-be-called"

        def generate(self, *a, **k):
            raise AssertionError("generator must not run for uncovered questions")

    result = ask(db, FakeEmbedder(768), ExplodingGenerator(), fake_settings, "Who won the 1998 World Cup final?")
    assert not result.answered and result.text == NOT_IN_CORPUS
    rec = json.loads(fake_settings.log_path.read_text().splitlines()[-1])
    assert rec["answered"] is False and rec["grounded"] is False


def test_source_filter_restricts_results(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)
    result = ask(
        db, FakeEmbedder(768), FakeGenerator(), fake_settings, "What was the Kestrel-4 uptime?",
        source="kestrel_production_2026.csv", log=False,
    )
    assert result.chunks and {c.source for c in result.chunks} == {"kestrel_production_2026.csv"}
    assert "well: Kestrel-4" in result.text and "[kestrel_production_2026.csv row " in result.text


def test_keyword_only_hit_is_found_by_fts(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)
    hits = store.fts_search(db, "waterflood pilot", "fake-hash-embedding", 5)
    assert hits and hits[0].source == "kestrel_reserves_memo.pdf" and hits[0].page == 2


# ---- second-pass fixes ----
def _strict_lines(path):
    def reject(name):
        raise AssertionError(f"non-JSON constant {name} in request log")

    return [json.loads(line, parse_constant=reject) for line in path.read_text().splitlines()]


class _FixedGenerator:
    model = "stub"

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, question, chunks):
        from ragdemo.generator import GenerationResult

        self.calls += 1
        return GenerationResult(self.text, 10, 5, 0)


def test_term_guard_refuses_a_month_the_corpus_does_not_have(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)
    gen = _FixedGenerator("should not be asked")
    # On topic (high similarity) but the corpus only has January to March.
    result = ask(db, FakeEmbedder(768), gen, fake_settings, "What was the uptime of Kestrel-4 in June 2026?")
    assert not result.answered and result.text == NOT_IN_CORPUS and gen.calls == 0
    rec = _strict_lines(fake_settings.log_path)[-1]
    assert rec["refusal_reason"] == "unsupported_terms" and "june" in rec["unsupported_terms"]
    # The same question about a month that exists goes through to the generator.
    gen = _FixedGenerator("Kestrel-4 uptime was 95.9 [kestrel_production_2026.csv row 3]")
    ok = ask(db, FakeEmbedder(768), gen, fake_settings, "What was the uptime of Kestrel-4 in March 2026?")
    assert ok.answered and gen.calls == 1


def test_model_refusal_sentinel_is_enforced(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)
    gen = _FixedGenerator("Hmm. " + NOT_IN_CORPUS.lower())
    result = ask(db, FakeEmbedder(768), gen, fake_settings, "What are the 2P reserves of Kestrel Field?")
    assert gen.calls == 1 and not result.answered and result.text == NOT_IN_CORPUS
    rec = _strict_lines(fake_settings.log_path)[-1]
    assert rec["grounded"] is True and rec["answered"] is False and rec["refusal_reason"] == "model_refusal"


def test_term_guard_can_be_disabled(db, corpus_dir, fake_settings):
    from dataclasses import replace

    _ingest(db, corpus_dir, fake_settings)
    gen = _FixedGenerator("An answer [kestrel_production_2026.csv row 1]")
    off = replace(fake_settings, term_guard=False)
    assert ask(db, FakeEmbedder(768), gen, off, "What was the uptime of Kestrel-4 in June 2026?", log=False).answered


def test_offline_answer_for_2p_reserves_picks_the_right_sentence(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)
    result = ask(db, FakeEmbedder(768), FakeGenerator(), fake_settings, "What are the 2P reserves of Kestrel Field?")
    assert "21.7 MMbbl" in result.text and "[kestrel_reserves_memo.pdf p.1]" in result.text


def test_fake_mode_reports_null_tokens_per_sec(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)
    ask(db, FakeEmbedder(768), FakeGenerator(), fake_settings, "What are the 2P reserves of Kestrel Field?")
    assert _strict_lines(fake_settings.log_path)[-1]["tokens_per_sec"] is None


def test_zero_vector_question_logs_strict_json(db, corpus_dir, fake_settings):
    _ingest(db, corpus_dir, fake_settings)
    # Only stopwords: the fake embedder returns a zero vector, whose cosine distance is NaN.
    result = ask(db, FakeEmbedder(768), FakeGenerator(), fake_settings, "What is it?")
    assert not result.answered
    ask(db, FakeEmbedder(768), FakeGenerator(), fake_settings, "What are the 2P reserves of Kestrel Field?")
    records = _strict_lines(fake_settings.log_path)  # raises on any NaN/Infinity token
    assert len(records) == 2 and records[0]["scores"]["top_similarity"] is None and records[0]["scores"]["vector"] == []


def test_zero_vector_chunks_are_not_stored(db, fake_settings):
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "blank.txt").write_text("the and of to")  # only stopwords -> zero embedding
        (Path(tmp) / "real.txt").write_text("Kestrel reserves are large.")
        report = ingest_path(db, FakeEmbedder(768), Path(tmp), batch_size=4)
    assert report.inserted == 1 and any("zero embedding" in w for w in report.warnings)


def test_ask_before_migrate_gives_a_clear_message(empty_db_url, monkeypatch, capsys):
    from ragdemo import cli, config

    monkeypatch.setattr(config, "load_env_file", lambda path: None)  # never touch the real .env.local in tests
    monkeypatch.setenv("DATABASE_URL", empty_db_url)
    code = cli.main(["--fake", "ask", "What are the 2P reserves?"])
    err = capsys.readouterr().err
    assert code == 2 and "no schema" in err and "migrate" in err and "UndefinedTable" not in err
    assert cli.main(["--fake", "migrate"]) == 0
    assert cli.main(["--fake", "ask", "What are the 2P reserves?"]) == 0  # works once migrated (empty corpus refuses)
