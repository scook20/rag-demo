# rag-demo

[![ci](https://github.com/scook20/rag-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/scook20/rag-demo/actions/workflows/ci.yml)

A demo retrieval-augmented generation (RAG) pipeline written in Python. It ingests PDF, Markdown/text and CSV files, embeds them with Gemini, stores them in Postgres with pgvector, retrieves with hybrid search (vector plus full-text, fused by reciprocal rank fusion), and answers with inline citations such as `[report.pdf p.4]`. When the corpus does not cover a question it says so instead of guessing (three layers: a similarity gate, a check that the question's specific terms appear in the retrieved text, and the model's own refusal), and every question writes a JSON metrics line with latency, tokens, estimated cost and the chunks used. It is a demo built to show the architecture, not a production system, and it is designed so that moving from a laptop to Google Cloud is a configuration change.

## At a glance

I designed and built this alone, from model selection to the eval harness: about 3,000 lines of Python and SQL, with 67 tests that run offline without an API key.

| Area | What it shows |
|---|---|
| **LLM orchestration** | Batched embedding with retry, backoff and the server's `retryDelay` honoured; cited generation; a fail-fast path when a daily quota runs out |
| **Guardrails** | A three-layer "not in corpus" gate (similarity, term coverage, model refusal) that refuses before spending a model call where it can |
| **Data layer** | Postgres 18 with pgvector (HNSW cosine) plus a GIN full-text index, SQL migrations, and content-hash idempotent upserts keyed on (hash, model) so re-ingesting is free and vector spaces never mix |
| **Observability** | One strict-JSON metrics line per request: latency, tokens, cost estimate, scores, chunk ids and refusal reason |
| **Evaluation** | 32 labeled questions; recall@k per retriever, refusal accuracy, citation validity, an optional LLM judge and a guards-off ablation. Live result: hybrid recall@5 of 88.6%, with 0 of 22 answerable questions wrongly blocked |
| **Cloud portability** | The same code targets the Gemini API or Vertex AI through environment variables; plain Postgres moves to Cloud SQL |
| **Honest measurement** | The eval found two real weaknesses (similarity can't separate on-topic unanswerable questions; RRF dropped two keyword hits). Both are written up below rather than tuned away |

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
        F[PDF / MD / TXT / CSV<br/>data/raw/] --> L[Loaders<br/>page numbers, CSV rows]
        L --> C[Chunker<br/>overlap + content hash]
        C --> K{hash already<br/>stored?}
        K -- no --> E[Embedder<br/>batched, retry + backoff]
        K -- yes --> S0[skip]
        E --> U[(Postgres + pgvector<br/>HNSW cosine + GIN tsvector)]
    end
    subgraph Ask
        Q[Question] --> QE[Embed query]
        QE --> V[Vector top-k<br/>cosine]
        Q --> T[Full-text top-k<br/>tsvector]
        U --> V
        U --> T
        V --> R[Reciprocal rank fusion]
        T --> R
        R --> G{best cosine >= threshold<br/>and question terms found<br/>in the excerpts?}
        G -- no --> N[Corpus does not cover this]
        G -- yes --> M[Gemini generation<br/>cited answer or refusal]
        M --> A[Answer + metrics line<br/>logs/requests.jsonl]
        N --> A
    end
```

## Run it

Prerequisites: Docker, and Python 3.11 or newer (developed on 3.13).

```bash
cd rag-demo

# 1. Start Postgres 18 with pgvector 0.8.7 on localhost:5433
docker compose up -d

# 2. Create a virtual environment and install pinned dependencies
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

# 3. Configure. Copy the example, then put your key in .env.local (gitignored)
cp .env.example .env.local        # edit GEMINI_API_KEY

# 4. Ingest documents (PDF, .md, .txt, .csv) and ask questions (ingest migrates first;
#    `ask` before any migration stops with a message telling you to run `migrate`)
python -m ragdemo migrate
python -m ragdemo ingest data/raw/
python -m ragdemo ask "What are the 2P reserves?"
python -m ragdemo ask "What failed on Kestrel-4?" --source kestrel_operations_notes.md

# Measure it (writes evals/reports/eval-<timestamp>-<mode>.{json,md})
python -m ragdemo eval                     # live: retrieval, guards, answers, citations
python -m ragdemo eval --retrieval-only    # live embeddings only, never calls the generation model
python -m ragdemo eval --judge --ablation  # plus LLM-judge faithfulness and a guards-off comparison

# Shut down when finished
docker compose down
```

No documents ship in `data/raw/`. A tiny fictional sample corpus (an invented oil field) lives in `tests/fixtures/corpus/`, so you can try the pipeline immediately with `python -m ragdemo ingest tests/fixtures/corpus/`.

Offline mode needs no API key. It swaps in a deterministic hashed bag-of-words embedder and an extractive generator: `python -m ragdemo --fake ingest tests/fixtures/corpus/` then `python -m ragdemo --fake ask "..."`. Fake vectors are stored under their own model name, so they never mix with real ones.

Run the tests (no API key needed; the integration tests skip when the database is down):

```bash
pytest
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | none | Gemini Developer API key |
| `DATABASE_URL` | `postgresql://ragdemo:ragdemo@localhost:5433/ragdemo` | Postgres connection |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `GENERATION_MODEL` | `gemini-3.8-flash` | Answer model |
| `GOOGLE_GENAI_USE_VERTEXAI` | unset | Set to `true` to use Vertex AI (also needs `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) |
| `MIN_SIMILARITY` | `0.55` | Fail-closed threshold on the best cosine similarity (calibrated on live embeddings, see Evaluation) |
| `TERM_GUARD` | `1` | Second guard: refuse when a specific question term (name, number, month) is in none of the retrieved excerpts |
| `GENERATION_THINKING_LEVEL` | `low` | Gemini thinking level (`low`, `medium`, `high`; `minimal` is rejected by the model) |
| `TOP_K`, `CANDIDATES`, `RRF_K` | `5`, `20`, `60` | Final chunks, per-retriever pool, fusion constant |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `1000`, `150` | Characters |

Prices used for cost estimates live in `config/pricing.yaml` with a `verified_on` date and source URL.

## Design choices

**Hybrid search.** Vector search finds paraphrases but is weak on exact tokens such as well names, part numbers and acronyms. Full-text search is the opposite. Running both and fusing the rankings with reciprocal rank fusion (`score = sum of 1 / (60 + rank)`) needs no score calibration between two very different scales, and it keeps both strengths. The keyword side ORs the question's terms, because Postgres ANDs bare words by default and that returns nothing for most natural questions.

**Content-hash idempotency.** Each chunk is keyed by a SHA-256 over its source, page, row and text, plus the embedding model. Re-ingesting a file finds every hash already stored, embeds nothing and inserts nothing, so reruns are free and safe after a crash. Changing the embedding model creates a parallel set of rows rather than silently mixing vector spaces, and retrieval only searches the current model. Edited files add new chunks; removing stale ones is not handled yet.

**Fail-closed "not in corpus", in three layers.** (1) If the best cosine similarity among the candidates is below `MIN_SIMILARITY`, the pipeline refuses without calling the model. (2) A term guard then checks that every specific term in the question (capitalised names, numbers, acronyms, month names) appears in the retrieved excerpts, so "uptime of Kestrel-4 in June" is refused when the corpus only covers January to March, even though the question is on topic and scores a high similarity. (3) The model is told to reply with a fixed sentence when the excerpts lack the detail, and that sentence is detected and normalised in code. Layers 1 and 2 also save the model call. Similarity alone cannot do this job: in the eval, on-topic questions the corpus cannot answer scored as high as answerable ones (see Evaluation).

**Cloud-portable by construction.** Models are called through the `google-genai` SDK, which targets the Gemini Developer API with an API key today and Vertex AI through environment variables later. The database is plain Postgres with the pgvector extension, which Cloud SQL for PostgreSQL supports. Schema changes are SQL files in `migrations/`, applied by a small runner.

More detail, including verified model IDs and sources, is in [docs/decisions.md](docs/decisions.md).

## Metrics

Each `ask` appends one strict-JSON line to `logs/requests.jsonl` (non-finite numbers are written as `null`): retrieval and generation latency in milliseconds, tokens in and out (including thinking tokens), output tokens per second (null in offline mode), estimated cost, the top-k similarity and fusion scores, the chunk ids, why a question was refused (`refusal_reason`: `low_similarity`, `unsupported_terms` or `model_refusal`), and whether every citation points at a retrieved chunk. The Gemini embedding API does not return token counts (verified live), so the query token count is a rough estimate (characters / 4) and flagged `embed_in_estimated`.

## Evaluation

`python -m ragdemo eval` runs 32 labeled questions (22 answerable, 10 unanswerable, in `evals/questions.yaml`) through the real pipeline and writes a timestamped JSON and Markdown report to `evals/reports/`. Metrics: retrieval recall@1/3/5 (the labeled evidence chunk is in the top k, scored for hybrid, vector-only and keyword-only), refusal accuracy on unanswerable questions, false refusals on answerable ones, citation validity (every citation points at a retrieved chunk) and whether a citation names the right source and page, answer correctness by key-fact match, and an optional LLM-judge faithfulness score (`--judge`, same model family as the generator, so biased upward). `--ablation` reruns the unanswerable questions with both guards off to show what the model does alone. The report also contains a similarity-threshold sweep, which is how `MIN_SIMILARITY` was calibrated.

### Results (live Gemini, 2026-10-21)

Retrieval and the pre-generation guards, measured with live `gemini-embedding-001` embeddings (report: [evals/reports/eval-20261021T210744Z-live-retrieval-only.md](evals/reports/eval-20261021T210744Z-live-retrieval-only.md)):

| Metric (24 chunks, 22 answerable questions) | recall@1 | recall@3 | recall@5 |
|---|---|---|---|
| Hybrid (vector + keyword, RRF) | 79.5% | 84.1% | 88.6% |
| Vector only | 75.0% | 88.6% | 88.6% |
| Keyword only | 79.5% | 84.1% | 88.6% |

| Guard measurement | Result |
|---|---|
| Unanswerable questions blocked before the model call (similarity gate + term guard) | 5 of 10 (2 off-topic by similarity, 3 near-miss by the term guard) |
| Answerable questions wrongly blocked before the model call | 0 of 22 |
| Best cosine similarity, answerable | min 0.570, median 0.710 |
| Best cosine similarity, unanswerable | median 0.718, max 0.799 |

Answer correctness, citation accuracy, model refusals and the judge score need the generation model across about 30 calls. The key used for this pass is on the free tier (20 requests per day per model), which was used up by verification, so the end-to-end live report is not in the repo yet. Run `python -m ragdemo eval --judge --ablation --resume` with billing enabled (about $0.05) or after the daily quota resets, and the report gains the answer metrics (`--resume` keeps finished questions across rate-limit stops). Live single-question runs did work end to end: see the transcript below.

What the eval showed that offline tests could not:

- **Similarity cannot separate answerable from unanswerable here.** Real embeddings of on-topic questions all land between 0.57 and 0.80 whether or not the corpus has the answer; "uptime of Kestrel-4 in June" scored 0.749, higher than the correct answers to several answerable questions. Only off-topic questions (0.47 and 0.55) fall low. Hence the term guard and the model refusal.
- **Hybrid fusion lost two answers that keyword search ranked first.** For "What was the flaring total in March 2026?" and the rod-pump question, the keyword retriever put the operations notes at rank 1, but the vector retriever ranked per-well CSV rows highly, and rows that appear mid-list in both retrievers outscore a document that is first in only one. Hybrid recall@5 therefore equals each retriever alone (88.6%) even though the two together cover all 22 questions. Candidate fixes: weighted fusion, a re-ranker, or larger top-k. Not done yet.
- **Five near-miss questions pass both guards** (oil price assumption, Kestrel-7 start date, waterflood cost, H2S concentration, tank battery headcount) because every term in them exists somewhere in the retrieved text. Only the model's refusal can catch these; that is the number the pending end-to-end run measures.

### Live transcript (gemini-3.8-flash, `thinking_level=low`)

```text
$ python -m ragdemo ask "What are the 2P reserves of Kestrel Field?"
The proved plus probable (2P) reserves for Kestrel Field are 21.7 MMbbl of oil [kestrel_reserves_memo.pdf p.1].
Retrieved:
  [kestrel_reserves_memo.pdf p.1] id=40 cosine=0.730 rrf=0.0328
  ...
answered=True top_similarity=0.7295 est_cost=$0.0008595

$ python -m ragdemo ask "What was the uptime of Kestrel-4 in June 2026?"
The corpus does not cover this question, so I can't answer it from the documents provided.
answered=False top_similarity=0.7488 est_cost=$1.65e-06      # refused by the term guard (june), no model call
```

### Caveats

The corpus is fictional and small (7 files, 24 chunks), so retrieval is easier than on a real document set. The 32 questions and their labels were written by the person who wrote the corpus, so they favour what the pipeline already does well. With 22 answerable questions one question is 4.5 points. The `MIN_SIMILARITY` calibration rests on a thin margin (lowest answerable 0.570, highest blocked off-topic 0.547) and should be redone for any other corpus: the sweep in each report gives the numbers. This is a regression and sanity check, not a benchmark.

## Limits of this demo

- Character-based chunking, no re-ranker, no query rewriting.
- The term guard is a surface-form heuristic: it refuses a question that rephrases a number ("ten ppm" for "10 ppm") and does not recognise lowercase names.
- Scanned PDFs without a text layer are skipped with a warning (no OCR).
- Answer quality is measured on a small, fictional, builder-labeled set (see Caveats).
- Single user, local credentials, no authentication.

## Roadmap

1. **Finish the live end-to-end eval** with billing enabled, then try weighted fusion or a re-ranker and re-measure.
2. **Public corpus.** Re-run the eval on a few public documents with labels written independently of the pipeline.
3. **Deploy on Google Cloud.** Cloud SQL for PostgreSQL with pgvector, Vertex AI for models (`GOOGLE_GENAI_USE_VERTEXAI=true`), and the CLI exposed as a service on Cloud Run.
4. **Agent layer.** An ADK agent that uses retrieval as a tool.
5. Stale-chunk cleanup and an optional UI.
