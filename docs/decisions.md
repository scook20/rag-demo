---
type: decision-log
date: 2026-10-21
topics: [rag, pgvector, gemini, hybrid-search, google-cloud, python]
status: active
---

# rag-demo decisions

Key choices for the v1 core (first pass) and the live verification pass, with the sources checked on 2026-10-21. Model IDs, SDK behavior and pricing change often: re-verify before relying on them.

## Models

| Decision | Choice | Why | Source |
|---|---|---|---|
| Embedding model | `gemini-embedding-001`, 768 dimensions | Listed as Stable. Accepts a list of strings and returns one vector per string (batching), supports `task_type` (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`), text only. 768 is a recommended size and fits under the 2000-dimension HNSW limit for `vector`. | https://ai.google.dev/gemini-api/docs/embeddings and https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2 (lists 001 as "Stable", "remains available" for text-only) |
| Not chosen | `gemini-embedding-2` | Also listed as Stable (page updated April 2026; the models overview page still shows `gemini-embedding-2-preview`, which is out of date). It has no `task_type`, uses prompt prefixes instead, and "produces a single aggregated embedding for multiple inputs", which breaks simple batched text embedding. Its embedding space is incompatible with 001. Revisit in the eval pass. | https://ai.google.dev/gemini-api/docs/embeddings |
| Generation model | `gemini-3.8-flash` | Re-checked 2026-10-21: the models page (updated 2026-10-21 UTC) labels it "New Stable" and `gemini-3.6-flash` "Stable". Accepted by the live API. Thinking levels low/medium/high, default medium; `minimal` is rejected. The pipeline sets `low` through `GENERATION_THINKING_LEVEL` to keep answers fast and cheap. | https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash and https://ai.google.dev/gemini-api/docs/models |
| Normalization | L2-normalize every vector client-side | Only 3072-dimension `gemini-embedding-001` vectors are pre-normalized; truncated sizes must be normalized manually. | https://ai.google.dev/gemini-api/docs/embeddings |
| SDK | `google-genai` 2.32.0, `genai.Client()` with no arguments | The client picks the Gemini Developer API (`GEMINI_API_KEY`) or Vertex AI from the environment. Both `GOOGLE_GENAI_USE_VERTEXAI` and `GOOGLE_GENAI_USE_ENTERPRISE` are read by the installed SDK (the latter wins if both conflict), plus `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. | https://github.com/googleapis/python-genai and the installed package source (`google/genai/_api_client.py`) |
| Embedding page | Re-checked 2026-10-21 (page footer 2026-10-09 UTC) | `gemini-embedding-001` accepts 128 to 3072 dimensions (768, 1536, 3072 recommended), 2,048 input tokens, text only, eight task types including `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY`. The page says nothing about token counts or a batch limit (the live API does: see below). | https://ai.google.dev/gemini-api/docs/embeddings |
| API surface | `client.models.generate_content` | Still supported in the SDK. Google's newest text docs show `client.interactions.create`; migrating is a candidate for the next pass. | https://github.com/googleapis/python-genai |
| Temperature | Not set | Docs state Gemini 3.6+ models ignore custom `temperature`, `top_p`, `top_k`. | https://ai.google.dev/gemini-api/docs/text-generation |

## Pricing (config/pricing.yaml)

- `gemini-3.8-flash`: $0.75 input and $3.75 output per 1M tokens through 2026-12-31 (output includes thinking tokens), rising to $1.50 and $7.50 on 2027-01-01. Source: https://ai.google.dev/gemini-api/docs/pricing (page "last updated 2026-10-19 UTC").
- `gemini-embedding-001`: not listed on the pricing page. The $0.15 per 1M tokens in the config is the embedding price quoted on that page for File Search, so treat it as an estimate (flagged `estimate: true`).
- `gemini-embedding-2`: $0.20 per 1M text input tokens (listed, unused in v1).

## Postgres and pgvector

| Decision | Choice | Source |
|---|---|---|
| Image | `pgvector/pgvector:0.8.7-pg18` (pgvector 0.8.7, latest release; the image tag was published 2026-10-01) | https://github.com/pgvector/pgvector and https://hub.docker.com/r/pgvector/pgvector/tags |
| Vector index | `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)` (the defaults, stated explicitly) | https://github.com/pgvector/pgvector |
| Distance | `<=>` is cosine distance; similarity is `1 - distance`; `SET LOCAL hnsw.ef_search` raises recall (default 40) | https://github.com/pgvector/pgvector |
| Filtered search | HNSW applies filters after the scan, so a selective `source` filter can return fewer rows than asked. The query enables `SET LOCAL hnsw.iterative_scan = relaxed_order` (pgvector 0.8.0+) when a filter is present, and re-sorts by similarity in Python. Confirm the Cloud SQL pgvector version supports it before deploying. | https://github.com/pgvector/pgvector |
| Dimension limit | `vector` HNSW supports up to 2000 dimensions (`halfvec` up to 4000), so 768 works and the default 3072 would not index | https://github.com/pgvector/pgvector |
| Full text | Stored generated column `to_tsvector('english'::regconfig, content)` with a GIN index (explicit `regconfig` is required for the expression to be immutable); `ts_rank_cd(..., 32)` for ranking | https://www.postgresql.org/docs/current/ddl-generated-columns.html and https://www.postgresql.org/docs/current/textsearch-controls.html |
| Query parsing | `websearch_to_tsquery` ANDs bare words and takes `or` as OR, so the question's words are joined with `or` to favor recall | https://www.postgresql.org/docs/current/textsearch-controls.html |
| Vector parameters | Passed as text (`'[0.1,0.2]'::vector`) so no extra Python package or numpy is needed | |

## Pipeline design

- **Idempotency key:** `UNIQUE (content_hash, embedding_model)` with `content_hash = sha256(source, page, row, text)`. Re-ingesting finds every hash, embeds nothing, inserts nothing. Locator fields are in the hash so identical text in two files stays two rows. A changed model gets its own rows.
- **Known gap:** an edited file adds new chunks but does not remove the old ones. Cleanup by source is a roadmap item.
- **Hybrid fusion:** reciprocal rank fusion with k = 60 over a 20-candidate pool from each retriever; final top 5.
- **Fail-closed gate:** compares the best cosine similarity among vector candidates to `MIN_SIMILARITY` (RRF scores are rank-based and cannot indicate absolute relevance), then applies the term guard, then the model sentinel (see Second-pass fixes). Default 0.55 for `gemini-embedding-001`, calibrated live (see Calibration); the fake embedder uses 0.15. Below the threshold or with an unsupported term the generator is never called.
- **Chunking:** character windows (1000, overlap 150) that prefer paragraph, line and sentence boundaries. PDFs are chunked per page so page numbers survive. CSV rows are one chunk each and keep their columns as JSONB.
- **Retries:** exponential backoff with jitter (up to 6 attempts, 1 s base, 30 s cap) on HTTP 429/408, 5xx and transport errors only.
- **Token counts:** generation reports `usage_metadata` (prompt, candidates, thoughts). Embedding token counts are read from `embedding.statistics.token_count` when the API returns it; otherwise the query count is estimated at characters / 4 and flagged.
- **Offline fakes:** a hashed bag-of-words embedder and an extractive generator sit behind the same interfaces, so the whole test suite and a full ingest and ask run need no key.
- **Python:** a 3.13 virtual environment (the host default 3.15 was not used); dependencies are pinned in `requirements.txt`.

## Live verification (2026-10-21)

Run against the real Gemini Developer API with the key in `.env.local` (never printed or logged), using fixtures only.

| Assumption from the first pass | Result | Detail |
|---|---|---|
| `gemini-3.8-flash` is a valid generation model ID | Confirmed | Answered `generate_content` calls; the first answered question used 827 prompt tokens. |
| `ThinkingConfig(thinking_level="low")` is accepted | Confirmed | `low` works. `minimal` is rejected: `400 INVALID_ARGUMENT, Thinking level MINIMAL is not supported for this model`, as the docs say. With `low` the response reported no thinking tokens, so output tokens equal answer tokens. `medium` returned a transient 503 and was not tested. |
| Embedding calls return token counts | **No.** | `embed_content` on the Developer API returns `statistics=None` for every embedding, and no usage metadata. Query tokens are estimated as characters / 4 and flagged `embed_in_estimated: true`; ingest reports `embed_tokens=n/a`. Embedding cost is therefore an estimate. |
| A 100-item embedding batch works | Confirmed, and 100 is the hard limit | 100 items returned 100 vectors; 101 and 250 failed with `400: BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch`. `EMBED_BATCH_SIZE` defaults to 100. |
| 768-dimension vectors are not unit length | Confirmed | First vector norm 0.59; the client normalises every vector. |
| Retry on transient errors is needed | Confirmed | `gemini-3.8-flash` returned `503 UNAVAILABLE: high demand` on several consecutive calls the same day. The retry loop absorbed it, at the cost of one answer taking 76 s and another 136 s. |

### Free-tier quota (finding)

The key is on the Gemini API free tier. `gemini-3.8-flash` allows 20 requests per day per project (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, limit 20, reset in about 10 hours); `gemini-3.6-flash` allows 5 per minute and also ran out of daily quota after about 20 calls. Failed 503 attempts appear to count. Consequences: (1) the end-to-end live eval (about 30 generation calls, 60 or more with `--judge --ablation`) cannot finish on the free tier in one day, so only the retrieval-only live report is committed; (2) free-tier prompts and responses may be used by Google to improve its products (pricing page), so only fictional fixtures were sent, and real client or confidential documents must not go through this key until billing is enabled; (3) `with_retries` now reads the server's `retryDelay`: a per-minute limit is waited out, a delay over 120 s (a daily quota) fails fast instead of sleeping for hours. Rate limits per model are in AI Studio (https://aistudio.google.com/rate-limit); the docs page lists no numbers (https://ai.google.dev/gemini-api/docs/rate-limits, updated 2026-10-15 UTC). Free tier was verified at https://ai.google.dev/gemini-api/docs/pricing.

### Pricing re-check (2026-10-21)

https://ai.google.dev/gemini-api/docs/pricing (updated 2026-10-19 UTC): `gemini-3.8-flash` and `gemini-3.6-flash` both $0.75 input and $3.75 output (incl. thinking) per 1M tokens through 2026-12-31, then $1.50 and $7.50 from 2027-01-01. Only `gemini-embedding-2` ($0.20 per 1M text tokens) is listed for embeddings; `gemini-embedding-001` still has no listed price, so $0.15 stays an estimate. Actual billed spend for this pass was $0 (free tier); the list-price equivalent is under $0.01.

## Second-pass fixes (verifier findings)

| Finding | Fix | Test |
|---|---|---|
| (a) Gate passes partly related questions | Three layers (`answer.py`, `guard.py`, `generator.py`): similarity gate, term guard (specific question terms must appear in the retrieved text; month names also match ISO dates), model sentinel detected tolerantly by `is_refusal` and normalised. The prompt now says to refuse when the subject is present but the asked detail is not. Every refusal logs `refusal_reason`. | `test_term_guard_refuses_a_month_the_corpus_does_not_have`, `test_model_refusal_sentinel_is_enforced`, `tests/test_guard.py` |
| (b) Offline generator picks the wrong sentence for "2P reserves" | `FakeGenerator` scores sentences with inverse-sentence-frequency weights, triple weight for tokens with digits, across all retrieved chunks. | `test_fake_generator_prefers_the_distinctive_word_over_shared_words`, `test_offline_answer_for_2p_reserves_picks_the_right_sentence` |
| (c) NaN similarity from zero vectors writes invalid JSON | A zero query vector skips vector search; NaN rows are dropped in `store.vector_search`; zero-embedding chunks are skipped at ingest with a warning; `append_jsonl` converts any non-finite float to null and sets `allow_nan=False`. | `test_zero_vector_question_logs_strict_json`, `test_strict_json_replaces_nan_and_infinity_with_null`, `test_zero_vector_chunks_are_not_stored` (every log line parsed with a constant-rejecting JSON parser) |
| (d) `ask` before `migrate` raises `UndefinedTable` | `store.schema_ready` check; the CLI prints "the database has no schema yet. Run `python -m ragdemo migrate`..." and exits 2. Chosen over auto-migrating because `ask` is a read command. | `test_ask_before_migrate_gives_a_clear_message` |
| (e) Absurd `tokens_per_sec` in fake mode | Generators carry `synthetic = True` (fake); `tokens_per_sec` is null for them. | `test_fake_mode_reports_null_tokens_per_sec` |

## Calibration of MIN_SIMILARITY

Method: `python -m ragdemo eval --retrieval-only` on live embeddings records the best cosine similarity for every question and sweeps thresholds (report table). Result: answerable questions score 0.570 to 0.789; on-topic unanswerable questions score 0.624 to 0.799; off-topic ones 0.471 and 0.547. No threshold separates the on-topic classes, so similarity stays a coarse gate for off-topic input. Chosen default **0.55**: the highest round value that keeps 100% of the answerable questions (lowest is 0.570) while blocking both off-topic questions. The margin is thin and corpus specific (see README caveats). The offline fake embedder keeps 0.15.

## Eval design

- **Labels** (`evals/questions.yaml`): evidence is `{source, page?, contains}` (a substring of the chunk text), so labels survive re-chunking; key facts are lists of acceptable spellings, with thousands separators normalised. Loader rejects answerable questions without evidence or key facts, and a test checks every evidence label against the real fixture corpus.
- **Corpus:** the three first-pass fixtures plus four more fictional files (HSE procedures, well tests, facilities study, company overview) to reach 24 chunks. All are marked fictional.
- **Metrics:** recall@1/3/5 per retriever; refusal accuracy and false refusals; citation validity and gold-source citation; key-fact answer correctness; optional LLM judge (same model family, labeled as biased); optional guards-off ablation; threshold sweep. Questions that hit an API error after retries are recorded and excluded from scores, and `--resume` checkpoints finished questions so a rate-limited key can finish over several runs.
- **Reports:** `evals/reports/eval-<UTC timestamp>-<live|live-retrieval-only|offline>.{json,md}`. Live reports are committed; offline reports are gitignored. Reports contain questions, answers and scores only, never credentials.
- **Offline mode** uses the hashed bag-of-words embedder and extractive generator. It checks that the harness and metrics work; its numbers say nothing about Gemini.
