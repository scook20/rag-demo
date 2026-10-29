-- Core schema for the RAG demo. Applied once by the migration runner (src/ragdemo/store.py).
-- Vector dimension 768 must match EMBEDDING_DIM. HNSW on `vector` supports up to 2000 dims.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id              BIGSERIAL PRIMARY KEY,
    content_hash    TEXT        NOT NULL,   -- sha256 over source, page, row and content
    embedding_model TEXT        NOT NULL,   -- vectors from different models never mix
    source          TEXT        NOT NULL,   -- path relative to the ingest root
    page            INTEGER,                -- 1-based PDF page, NULL for md/txt/csv
    chunk_index     INTEGER     NOT NULL,   -- position within the source (or page)
    content         TEXT        NOT NULL,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- CSV columns, file type, row number
    embedding       vector(768) NOT NULL,
    content_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunks_hash_model_key UNIQUE (content_hash, embedding_model)
);

-- Approximate nearest neighbour, cosine distance (operator <=>).
CREATE INDEX chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Keyword search.
CREATE INDEX chunks_content_tsv_idx ON chunks USING gin (content_tsv);

-- Metadata filter by source.
CREATE INDEX chunks_source_idx ON chunks (source);
