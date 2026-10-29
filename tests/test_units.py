import json
import math

import pytest
from google.genai import errors

from ragdemo.config import load_env_file
from ragdemo.embedder import FakeEmbedder, GeminiEmbedder, l2_normalize
from ragdemo.generator import NOT_IN_CORPUS, FakeGenerator, build_prompt, is_refusal
from ragdemo.metrics import append_jsonl, dumps_strict, estimate_cost, load_pricing
from ragdemo.models import RetrievedChunk, citation_label
from ragdemo.config import PROJECT_ROOT
from ragdemo.retrieve import rrf_fuse
from ragdemo.retry import is_retryable, server_retry_delay, with_retries
from ragdemo.store import keyword_query, vector_literal


def chunk(i, **kw):
    return RetrievedChunk(id=i, source="s.md", page=None, chunk_index=0, content=f"c{i}", metadata={}, **kw)


# ---- rank fusion ----
def test_rrf_scores_and_order():
    vec = [chunk(1, vector_similarity=0.9, vector_rank=1), chunk(2, vector_similarity=0.8, vector_rank=2)]
    fts = [chunk(2, fts_score=0.5, fts_rank=1), chunk(3, fts_score=0.2, fts_rank=2)]
    fused = rrf_fuse(vec, fts, k=60)
    by_id = {c.id: c for c in fused}
    assert [c.id for c in fused] == [2, 1, 3]  # id 2 appears in both lists
    assert by_id[2].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[2].vector_similarity == 0.8 and by_id[2].fts_score == 0.5  # details merged
    assert by_id[3].vector_similarity is None


def test_rrf_limit_and_empty():
    assert rrf_fuse([], []) == []
    assert len(rrf_fuse([chunk(i, vector_rank=i) for i in range(1, 6)], [], limit=2)) == 2


# ---- retry ----
def test_retry_succeeds_after_transient_errors():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise errors.ServerError(503, {"error": {"message": "unavailable"}})
        return "ok"

    assert with_retries(flaky, sleep=sleeps.append, base_delay=1.0) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0  # exponential backoff with jitter


def test_retry_gives_up_and_skips_non_retryable():
    def always_429():
        raise errors.ClientError(429, {"error": {"message": "rate limited"}})

    with pytest.raises(errors.ClientError):
        with_retries(always_429, max_attempts=3, sleep=lambda s: None)

    calls = []

    def bad_request():
        calls.append(1)
        raise errors.ClientError(400, {"error": {"message": "bad"}})

    with pytest.raises(errors.ClientError):
        with_retries(bad_request, sleep=lambda s: None)
    assert len(calls) == 1
    assert is_retryable(errors.ClientError(429, {})) and not is_retryable(ValueError("x"))


# ---- embedder ----
def test_fake_embedder_is_deterministic_unit_and_semantic():
    emb = FakeEmbedder(dim=768)
    a = emb.embed_documents(["Kestrel proved reserves are 14.2 MMbbl"]).vectors[0]
    assert a == emb.embed_documents(["Kestrel proved reserves are 14.2 MMbbl"]).vectors[0]
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    q_rel = emb.embed_query("What are the proved reserves?").vectors[0]
    q_irrel = emb.embed_query("capital of France").vectors[0]
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert dot(a, q_rel) > 0.3 > abs(dot(a, q_irrel))


def test_l2_normalize():
    v = l2_normalize([3.0, 4.0])
    assert v == pytest.approx([0.6, 0.8])
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


class _FakeGenaiClient:
    """Stands in for genai.Client to test batching and normalisation without the network."""

    def __init__(self):
        self.calls = []
        self.models = self

    def embed_content(self, model, contents, config):
        from google.genai import types

        self.calls.append((model, list(contents), config.task_type, config.output_dimensionality))
        embeddings = [
            types.ContentEmbedding(values=[2.0, 0.0], statistics=types.ContentEmbeddingStatistics(token_count=5))
            for _ in contents
        ]
        return types.EmbedContentResponse(embeddings=embeddings)


def test_gemini_embedder_batches_normalises_and_sums_tokens():
    client = _FakeGenaiClient()
    emb = GeminiEmbedder("gemini-embedding-001", dim=2, batch_size=2, client=client)
    result = emb.embed_documents(["a", "b", "c"])
    assert [len(c[1]) for c in client.calls] == [2, 1]
    assert client.calls[0][2:] == ("RETRIEVAL_DOCUMENT", 2)
    assert result.vectors == [[1.0, 0.0]] * 3 and result.tokens == 15
    emb.embed_query("q")
    assert client.calls[-1][2] == "RETRIEVAL_QUERY"


def test_gemini_embedding_2_rejected():
    with pytest.raises(ValueError):
        GeminiEmbedder("gemini-embedding-2", dim=768, client=object())


# ---- generator and prompt ----
def test_prompt_labels_excerpts_and_fake_generator_cites():
    c = RetrievedChunk(1, "memo.pdf", 4, 0, "2P reserves are 21.7 MMbbl. More text.", {})
    prompt = build_prompt("What are 2P reserves?", [c])
    assert "Excerpt 1 [memo.pdf p.4]" in prompt and "Question: What are 2P reserves?" in prompt
    out = FakeGenerator().generate("q", [c])
    assert out.text == "2P reserves are 21.7 MMbbl. [memo.pdf p.4]"
    assert FakeGenerator().generate("q", []).text == NOT_IN_CORPUS


def test_fake_generator_picks_the_best_matching_sentence():
    c = RetrievedChunk(1, "n.md", None, 0, "Intro line. The pump failed on Kestrel-4. Closing line.", {"section": 2})
    assert FakeGenerator().generate("Which pump failed?", [c]).text == "The pump failed on Kestrel-4. [n.md section 2]"


def test_citation_labels():
    assert citation_label("a.pdf", 4, {}) == "a.pdf p.4"
    assert citation_label("p.csv", None, {"row": 3}) == "p.csv row 3"
    assert citation_label("n.md", None, {"section": 2}) == "n.md section 2"
    assert citation_label("n.md", None, {}) == "n.md"


# ---- store helpers ----
def test_keyword_query_ors_terms_and_drops_the_word_or():
    assert keyword_query("Which well had the ESP failure?") == "Which or well or had or the or ESP or failure"
    assert keyword_query("cats or dogs") == "cats or dogs"
    assert keyword_query("?!") == ""


def test_vector_literal():
    assert vector_literal([0.5, -1, 2.25]) == "[0.5,-1,2.25]"


# ---- metrics and config ----
def test_cost_estimate_uses_pricing_file():
    pricing = load_pricing(PROJECT_ROOT / "config" / "pricing.yaml")
    assert "verified_on" in pricing and pricing["source"].startswith("https://")
    cost = estimate_cost(pricing, "gemini-3.8-flash", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.75 + 3.75)
    assert estimate_cost(pricing, "unknown-model", 10, 10) is None
    assert estimate_cost(pricing, "fake-extractive", 100, 100) == 0.0


def test_append_jsonl_writes_one_line_per_call(tmp_path):
    path = tmp_path / "logs" / "r.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"b": "é"})
    assert path.read_text(encoding="utf-8").splitlines() == ['{"a":1}', '{"b":"é"}']


def test_env_file_loader_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env.local"
    f.write_text("# c\nRAGDEMO_T1=from_file\nRAGDEMO_T2='quoted'\nexport RAGDEMO_T3=3\n")
    monkeypatch.setenv("RAGDEMO_T1", "from_env")
    monkeypatch.delenv("RAGDEMO_T2", raising=False)
    monkeypatch.delenv("RAGDEMO_T3", raising=False)
    load_env_file(f)
    import os

    assert (os.environ["RAGDEMO_T1"], os.environ["RAGDEMO_T2"], os.environ["RAGDEMO_T3"]) == ("from_env", "quoted", "3")


# ---- second-pass fixes ----
def test_is_refusal_tolerates_formatting_and_extra_text():
    assert is_refusal(NOT_IN_CORPUS)
    assert is_refusal("  " + NOT_IN_CORPUS.upper().replace("CAN'T", "CAN’T") + "\n")
    assert is_refusal("Sorry. " + NOT_IN_CORPUS.rstrip(".") + " Please ask something else.")
    assert not is_refusal("The 2P reserves are 21.7 MMbbl [memo.pdf p.1]")


def test_fake_generator_prefers_the_distinctive_word_over_shared_words():
    c = RetrievedChunk(
        1, "memo.pdf", 1, 0,
        "Proved (1P) reserves for Kestrel Field are 14.2 MMbbl of oil.\n"
        "Proved plus probable (2P) reserves are 21.7 MMbbl of oil.\n"
        "Proved, probable and possible (3P) reserves are 30.1 MMbbl of oil.",
        {},
    )
    out = FakeGenerator().generate("What are the 2P reserves of Kestrel Field?", [c])
    assert out.text.startswith("Proved plus probable (2P) reserves are 21.7 MMbbl")


def test_fake_generator_can_pick_a_sentence_from_a_lower_ranked_chunk():
    a = RetrievedChunk(1, "a.md", None, 0, "Weather is mild.", {})
    b = RetrievedChunk(2, "b.md", None, 0, "The pump failed on Kestrel-4.", {})
    assert FakeGenerator().generate("Which pump failed?", [a, b]).text == "The pump failed on Kestrel-4. [b.md]"


def test_strict_json_replaces_nan_and_infinity_with_null():
    record = {"a": float("nan"), "b": [1.0, float("inf"), {"c": float("-inf")}], "d": 2.5}
    parsed = json.loads(dumps_strict(record), parse_constant=_reject_constant)
    assert parsed == {"a": None, "b": [1.0, None, {"c": None}], "d": 2.5}


def _reject_constant(name):
    raise AssertionError(f"non-JSON constant {name}")


def test_append_jsonl_never_writes_nan(tmp_path):
    path = tmp_path / "r.jsonl"
    append_jsonl(path, {"top_similarity": float("nan"), "scores": [float("nan"), 0.5]})
    line = path.read_text().strip()
    assert "NaN" not in line
    assert json.loads(line, parse_constant=_reject_constant) == {"top_similarity": None, "scores": [None, 0.5]}


def test_retry_honours_server_delay_and_gives_up_on_daily_quota():
    per_minute = errors.ClientError(429, {"error": {"message": "slow down", "details": [{"retryDelay": "24s"}]}})
    daily = errors.ClientError(429, {"error": {"message": "quota", "details": [{"retryDelay": "35712s"}]}})
    assert server_retry_delay(per_minute) == 24.0 and is_retryable(per_minute)
    assert not is_retryable(daily)  # waiting 10 hours is not a retry

    calls, sleeps = [], []

    def limited():
        calls.append(1)
        if len(calls) == 1:
            raise per_minute
        return "ok"

    assert with_retries(limited, sleep=sleeps.append, base_delay=1.0) == "ok"
    assert sleeps and sleeps[0] >= 25.0
