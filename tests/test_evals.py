import json
from pathlib import Path

import pytest

from ragdemo import cli, evals
from ragdemo.config import PROJECT_ROOT
from ragdemo.models import RetrievedChunk

QUESTIONS = PROJECT_ROOT / "evals" / "questions.yaml"


def chunk(i, source, content, page=None):
    return RetrievedChunk(i, source, page, 0, content, {})


def test_labels_load_and_are_valid():
    items = evals.load_questions(QUESTIONS)
    assert 20 <= len(items) <= 40
    assert any(not i.answerable for i in items) and any(i.answerable for i in items)
    assert {i.type for i in items} >= {"lookup", "paraphrase", "multi-evidence", "near-miss", "off-topic"}


def test_every_label_points_at_a_real_chunk_of_the_corpus(corpus_dir):
    from ragdemo.chunking import chunk_records
    from ragdemo.loaders import discover, load_file, source_name

    chunks = []
    for f in discover(corpus_dir):
        records, _ = load_file(f, source_name(f, corpus_dir))
        chunks += [chunk(0, c.source, c.content, c.page) for c in chunk_records(records)]
    for item in evals.load_questions(QUESTIONS):
        for e in item.evidence:
            assert evals.evidence_found(chunks, e), f"{item.id}: evidence not found in corpus: {e}"


def test_label_validation(tmp_path):
    bad = tmp_path / "q.yaml"
    bad.write_text("questions:\n  - {id: a, type: lookup, question: q, answerable: true}\n")
    with pytest.raises(ValueError):
        evals.load_questions(bad)
    bad.write_text("questions:\n  - {id: a, type: off-topic, question: q, answerable: false, key_facts: [[x]]}\n")
    with pytest.raises(ValueError):
        evals.load_questions(bad)


def test_evidence_recall_respects_source_page_and_text():
    chunks = [chunk(1, "a.pdf", "The 2P reserves", page=1), chunk(2, "a.pdf", "Water cut 38 percent", page=2)]
    ev = [{"source": "a.pdf", "page": 1, "contains": "2p reserves"}, {"source": "a.pdf", "page": 1, "contains": "water cut"}]
    assert evals.evidence_found(chunks, ev[0]) and not evals.evidence_found(chunks, ev[1])  # wrong page
    assert evals.evidence_recall(chunks, ev, k=2) == 0.5
    assert evals.evidence_recall(chunks, ev[:1], k=1) == 1.0
    assert evals.evidence_recall(chunks[1:], ev[:1], k=5) == 0.0


def test_key_facts_normalise_numbers_and_accept_alternatives():
    answer = "Kestrel-4 made 6,100 bbl after the ESP failure [kestrel_production_2026.csv row 6]"
    assert evals.key_fact_hits(answer, [["6100"], ["esp", "pump"], ["12000"]]) == [True, True, False]


def test_citation_gold_source_needs_matching_source_and_page():
    ev = [{"source": "a.pdf", "page": 2, "contains": "x"}]
    assert evals.cites_gold_source("claim [a.pdf p.2]", ev)
    assert not evals.cites_gold_source("claim [a.pdf p.1]", ev)
    assert not evals.cites_gold_source("claim [b.pdf p.2]", ev)
    assert evals.cites_gold_source("claim [x.csv row 6]", [{"source": "x.csv", "contains": "y"}])


def _result(**kw):
    base = {
        "id": "x", "type": "lookup", "question": "q", "answerable": True, "answered": True, "refusal_reason": None,
        "unsupported_terms": [], "top_similarity": 0.8, "answer": "a", "retrieved": [], "est_cost_usd": 0.001,
        "latency_ms": {"total": 10.0}, "citations": {"cited": [], "all_in_retrieved": None},
        "recall_hybrid": {"1": 1.0, "3": 1.0, "5": 1.0}, "recall_vector": {"1": 0.0, "3": 1.0, "5": 1.0},
        "recall_fts": {"1": 0.0, "3": 0.0, "5": 1.0}, "key_fact_hits": [True], "correct": True,
        "citation_valid": True, "citation_gold": True,
    }
    return {**base, **kw}


def test_summary_and_threshold_sweep():
    results = [
        _result(id="a"),
        _result(id="b", correct=False, key_fact_hits=[False], top_similarity=0.6),
        _result(id="c", answerable=False, answered=False, refusal_reason="low_similarity", top_similarity=0.4),
        _result(id="d", answerable=False, answered=True, top_similarity=0.7, answer="oops"),
    ]
    s = evals.summarize(results)
    assert s["answerable"] == 2 and s["unanswerable"] == 2
    assert s["answer_correctness"] == 0.5 and s["refusal_accuracy"] == 0.5 and s["false_refusal_rate"] == 0.0
    assert s["overall_answer_or_refuse_accuracy"] == 0.5
    assert s["recall_hybrid"]["5"] == 1.0 and s["recall_vector"]["1"] == 0.0
    assert s["unanswerable_outcomes"] == {"low_similarity": 1, "answered": 1}
    sweep = {r["threshold"]: r for r in evals.threshold_sweep(results)}
    assert sweep[0.5]["answerable_passing"] == 1.0 and sweep[0.5]["unanswerable_blocked"] == 0.5
    assert sweep[0.65]["answerable_passing"] == 0.5 and sweep[0.75]["unanswerable_blocked"] == 1.0
    assert "Per question" in evals.render_markdown(
        {"mode": "live", "generated_at": "t", "questions_file": "q", "corpus": {"path": "c", "chunks_in_db_for_model": 1},
         "config": {"embedding_model": "e", "embedding_dim": 1, "generation_model": "g", "thinking_level": "low",
                    "top_k": 5, "candidates": 20, "rrf_k": 60, "min_similarity": 0.5, "term_guard": True,
                    "chunk_size": 1000, "chunk_overlap": 150, "judge_model": None},
         "summary": s, "threshold_sweep": evals.threshold_sweep(results), "est_cost_usd": {"questions_and_judge": 0.0},
         "items": results}
    )


def test_offline_eval_end_to_end_writes_strict_json_and_markdown(empty_db_url, monkeypatch, tmp_path, capsys):
    from ragdemo import config

    monkeypatch.setattr(config, "load_env_file", lambda path: None)
    monkeypatch.setenv("DATABASE_URL", empty_db_url)
    out = tmp_path / "reports"
    assert cli.main(["--fake", "eval", "--out", str(out), "--ablation"]) == 0
    (json_path,) = out.glob("eval-*-offline.json")
    (md_path,) = out.glob("eval-*-offline.md")

    def reject(name):
        raise AssertionError(f"non-JSON constant {name}")

    report = json.loads(json_path.read_text(), parse_constant=reject)
    s = report["summary"]
    assert report["mode"] == "offline-fake" and s["questions"] == len(report["items"]) >= 20
    assert s["recall_hybrid"]["5"] is not None and s["refusal_accuracy"] is not None
    assert "Retrieval recall@5 (hybrid)" in md_path.read_text()
    assert "GEMINI_API_KEY" not in json_path.read_text()


def test_checkpoint_roundtrip_and_signature_guard(tmp_path):
    path = tmp_path / "cp.jsonl"
    sig = {"generation_model": "m", "min_similarity": 0.5}
    assert evals._load_checkpoint(path, sig) == {}
    evals._append_checkpoint(path, sig, {"id": "q01", "answered": True})
    evals._append_checkpoint(path, sig, {"id": "q02", "answered": False})
    assert set(evals._load_checkpoint(path, sig)) == {"q01", "q02"}
    with pytest.raises(ValueError):
        evals._load_checkpoint(path, {**sig, "generation_model": "other"})


def test_retrieval_only_eval_never_calls_the_model(db, corpus_dir, fake_settings):
    from ragdemo.embedder import FakeEmbedder
    from ragdemo.ingest import ingest_path

    ingest_path(db, FakeEmbedder(768), corpus_dir)

    class Exploding:
        model = "boom"

        def generate(self, *a, **k):
            raise AssertionError("model must not be called")

    items = evals.load_questions(QUESTIONS)[:6]
    results = evals.run_questions(db, FakeEmbedder(768), Exploding(), fake_settings, items, generate=False)
    summary = evals.summarize(results, generated=False)
    assert summary["generated"] is False and summary["answer_correctness"] is None
    assert summary["recall_hybrid"]["5"] is not None


def test_api_error_on_one_question_is_recorded_and_excluded(db, corpus_dir, fake_settings):
    from google.genai import errors

    from ragdemo.embedder import FakeEmbedder
    from ragdemo.ingest import ingest_path

    ingest_path(db, FakeEmbedder(768), corpus_dir)

    class QuotaGenerator:
        model = "quota"

        def generate(self, *a, **k):
            raise errors.ClientError(429, {"error": {"message": "quota", "details": [{"retryDelay": "35712s"}]}})

    items = evals.load_questions(QUESTIONS)[:3]
    results = evals.run_questions(db, FakeEmbedder(768), QuotaGenerator(), fake_settings, items)
    assert all("error" in r and "429" in r["error"] for r in results)
    summary = evals.summarize(results)
    assert summary["errored"] == 3 and summary["scored"] == 0 and summary["answer_correctness"] is None
