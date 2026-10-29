from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from ragdemo.config import DEFAULT_DATABASE_URL, PROJECT_ROOT, Settings

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture
def corpus_dir() -> Path:
    return FIXTURES


@pytest.fixture
def fake_settings(tmp_path) -> Settings:
    """Settings for the offline fakes, with a throwaway request log."""
    base = Settings.from_env(fake=True, load_files=False)
    return Settings(**{**base.__dict__, "log_path": tmp_path / "requests.jsonl", "migrations_dir": PROJECT_ROOT / "migrations"})


@pytest.fixture(scope="session")
def temp_db_url():
    """A throwaway database on the local pgvector container. Skips the test if the DB is down."""
    admin_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        admin = psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"pgvector database not reachable ({exc.__class__.__name__}); run `docker compose up -d`")
    name = f"ragdemo_test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    url = make_conninfo(admin_url, dbname=name)
    try:
        yield url
    finally:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


@pytest.fixture
def empty_db_url():
    """A separate, completely empty throwaway database (no schema). Skips if the DB is down."""
    admin_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        admin = psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"pgvector database not reachable ({exc.__class__.__name__})")
    name = f"ragdemo_empty_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    try:
        yield make_conninfo(admin_url, dbname=name)
    finally:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


@pytest.fixture
def db(temp_db_url, fake_settings):
    """A migrated connection with an empty chunks table."""
    from ragdemo import store

    conn = store.connect(temp_db_url)
    store.migrate(conn, fake_settings.migrations_dir)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE chunks RESTART IDENTITY")
    conn.commit()
    yield conn
    conn.close()
