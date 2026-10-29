# RAG eval report (live-retrieval-only)

Generated 2026-10-21T21:07:44+00:00. Labels: `evals/questions.yaml` (32 questions: 22 answerable, 10 unanswerable). Corpus: `tests/fixtures/corpus` (24 chunks).

Fictional corpus, small builder-authored label set: a regression and sanity check, not a benchmark.

## Configuration

Embedding `gemini-embedding-001` (768 dims), generation `gemini-3.8-flash` (thinking `low`), top_k 5, candidates 20, RRF k 60, min similarity 0.55, term guard on, chunks 1000/150.

## Results

| Metric | Value |
|---|---|
| Retrieval recall@1 (hybrid) | 79.5% |
| Retrieval recall@3 (hybrid) | 84.1% |
| Retrieval recall@5 (hybrid) | 88.6% |
| Retrieval recall@1 (vector only) | 75.0% |
| Retrieval recall@3 (vector only) | 88.6% |
| Retrieval recall@5 (vector only) | 88.6% |
| Retrieval recall@1 (keyword only) | 79.5% |
| Retrieval recall@3 (keyword only) | 84.1% |
| Retrieval recall@5 (keyword only) | 88.6% |
| Unanswerable blocked before the model (similarity gate + term guard) | 50.0% |
| Answerable wrongly blocked before the model | 0.0% |
| Answer correctness, citations, model refusals | not measured (retrieval-only run, the model was not called) |
| Latency p50 / max per question | 161.1 ms / 182.3 ms |
| Estimated spend (questions, judge, ablation) | $9e-06 |

How unanswerable questions were handled (`answered` = got past both guards and would reach the model): answered: 5, low_similarity: 2, unsupported_terms: 3


Top cosine similarity: answerable min 0.5697, median 0.7095; unanswerable median 0.718, max 0.7991.

## Similarity threshold sweep

Best-chunk cosine similarity as the only gate (term guard and model refusal not applied).

| Threshold | Answerable still answered | Unanswerable blocked |
|---|---|---|
| 0.30 | 100.0% | 0.0% |
| 0.35 | 100.0% | 0.0% |
| 0.40 | 100.0% | 0.0% |
| 0.45 | 100.0% | 0.0% |
| 0.50 | 100.0% | 10.0% |
| 0.55 | 100.0% | 20.0% |
| 0.60 | 95.5% | 20.0% |
| 0.65 | 72.7% | 30.0% |
| 0.70 | 59.1% | 40.0% |
| 0.75 | 31.8% | 80.0% |
| 0.80 | 0.0% | 100.0% |
| 0.85 | 0.0% | 100.0% |
| 0.90 | 0.0% | 100.0% |

## Per question

| Id | Type | Outcome | Top sim | Recall@5 | Detail |
|---|---|---|---|---|---|
| q01 | lookup | answerable: past guards | 0.730 | 100.0% |  |
| q02 | lookup | answerable: past guards | 0.768 | 100.0% |  |
| q03 | lookup | answerable: past guards | 0.750 | 100.0% |  |
| q04 | lookup | answerable: past guards | 0.754 | 100.0% |  |
| q05 | lookup | answerable: past guards | 0.709 | 100.0% |  |
| q06 | lookup | answerable: past guards | 0.710 | 100.0% |  |
| q07 | lookup | answerable: past guards | 0.719 | 0.0% |  |
| q08 | lookup | answerable: past guards | 0.683 | 100.0% |  |
| q09 | lookup | answerable: past guards | 0.783 | 100.0% |  |
| q10 | lookup | answerable: past guards | 0.768 | 100.0% |  |
| q11 | lookup | answerable: past guards | 0.789 | 100.0% |  |
| q12 | lookup | answerable: past guards | 0.618 | 100.0% |  |
| q13 | lookup | answerable: past guards | 0.635 | 100.0% |  |
| q14 | lookup | answerable: past guards | 0.706 | 100.0% |  |
| q15 | lookup | answerable: past guards | 0.682 | 100.0% |  |
| q16 | lookup | answerable: past guards | 0.764 | 100.0% |  |
| q17 | paraphrase | answerable: past guards | 0.570 | 0.0% |  |
| q18 | paraphrase | answerable: past guards | 0.650 | 100.0% |  |
| q19 | paraphrase | answerable: past guards | 0.626 | 100.0% |  |
| q20 | paraphrase | answerable: past guards | 0.630 | 100.0% |  |
| q21 | multi-evidence | answerable: past guards | 0.764 | 50.0% |  |
| q22 | multi-evidence | answerable: past guards | 0.675 | 100.0% |  |
| q23 | near-miss | unanswerable: blocked (unsupported_terms) | 0.749 | - | (june) |
| q24 | near-miss | unanswerable: blocked (unsupported_terms) | 0.750 | - | (executive) |
| q25 | near-miss | unanswerable: past guards | 0.661 | - |  |
| q26 | near-miss | unanswerable: past guards | 0.732 | - |  |
| q27 | near-miss | unanswerable: blocked (unsupported_terms) | 0.799 | - | (kestrel-5) |
| q28 | near-miss | unanswerable: past guards | 0.704 | - |  |
| q29 | near-miss | unanswerable: past guards | 0.758 | - |  |
| q30 | near-miss | unanswerable: past guards | 0.624 | - |  |
| q31 | off-topic | unanswerable: blocked (low_similarity) | 0.471 | - |  |
| q32 | off-topic | unanswerable: blocked (low_similarity) | 0.547 | - |  |
