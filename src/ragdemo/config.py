"""Settings from environment variables. A local .env.local file is loaded if present."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_URL = "postgresql://ragdemo:ragdemo@localhost:5433/ragdemo"
FAKE_EMBEDDING_MODEL = "fake-hash-embedding"
FAKE_GENERATION_MODEL = "fake-extractive"


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing variables."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_model: str
    generation_model: str
    embedding_dim: int
    thinking_level: str  # "" disables the thinking config
    use_fake: bool
    chunk_size: int
    chunk_overlap: int
    top_k: int
    candidates: int  # per-retriever candidate pool fed into rank fusion
    min_similarity: float  # fail-closed threshold on the best cosine similarity
    term_guard: bool  # second fail-closed guard: specific question terms must appear in the excerpts
    rrf_k: int
    embed_batch_size: int
    log_path: Path
    pricing_path: Path
    migrations_dir: Path

    @classmethod
    def from_env(cls, *, fake: bool | None = None, load_files: bool = True) -> "Settings":
        if load_files:
            load_env_file(PROJECT_ROOT / ".env.local")
        env = os.environ
        use_fake = _truthy(env.get("RAGDEMO_FAKE")) if fake is None else fake
        if use_fake:
            embedding_model = FAKE_EMBEDDING_MODEL
            generation_model = FAKE_GENERATION_MODEL
            default_threshold = "0.15"  # bag-of-words cosine runs lower than real embeddings
        else:
            embedding_model = env.get("EMBEDDING_MODEL", "gemini-embedding-001")
            generation_model = env.get("GENERATION_MODEL", "gemini-3.8-flash")
            default_threshold = "0.55"  # calibrated on live gemini-embedding-001, see docs/decisions.md
        return cls(
            database_url=env.get("DATABASE_URL", DEFAULT_DATABASE_URL),
            embedding_model=embedding_model,
            generation_model=generation_model,
            embedding_dim=int(env.get("EMBEDDING_DIM", "768")),
            thinking_level=env.get("GENERATION_THINKING_LEVEL", "low"),
            use_fake=use_fake,
            chunk_size=int(env.get("CHUNK_SIZE", "1000")),
            chunk_overlap=int(env.get("CHUNK_OVERLAP", "150")),
            top_k=int(env.get("TOP_K", "5")),
            candidates=int(env.get("CANDIDATES", "20")),
            min_similarity=float(env.get("MIN_SIMILARITY", default_threshold)),
            term_guard=_truthy(env.get("TERM_GUARD", "1")),
            rrf_k=int(env.get("RRF_K", "60")),
            embed_batch_size=int(env.get("EMBED_BATCH_SIZE", "100")),
            log_path=Path(env.get("REQUEST_LOG", str(PROJECT_ROOT / "logs" / "requests.jsonl"))),
            pricing_path=Path(env.get("PRICING_FILE", str(PROJECT_ROOT / "config" / "pricing.yaml"))),
            migrations_dir=Path(env.get("MIGRATIONS_DIR", str(PROJECT_ROOT / "migrations"))),
        )
