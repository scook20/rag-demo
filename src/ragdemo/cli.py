"""Command line: python -m ragdemo {migrate,ingest,ask}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragdemo import store
from ragdemo.answer import ask
from ragdemo.config import PROJECT_ROOT, Settings
from ragdemo.factory import build_embedder, build_generator
from ragdemo.ingest import ingest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragdemo", description="Demo RAG pipeline on Postgres + pgvector.")
    parser.add_argument("--fake", action="store_true", help="use the offline fake embedder and generator (no API key)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="apply SQL migrations")

    p_ingest = sub.add_parser("ingest", help="ingest a file or directory of PDF/MD/TXT/CSV files")
    p_ingest.add_argument("path", type=Path)

    p_ask = sub.add_parser("ask", help="ask a question against the ingested corpus")
    p_ask.add_argument("question")
    p_ask.add_argument("--source", help="only search this source path (as stored at ingest)")
    p_ask.add_argument("--top-k", type=int, help="number of chunks to pass to the model")

    p_eval = sub.add_parser("eval", help="run the labeled question set and write a report under evals/reports/")
    p_eval.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evals" / "questions.yaml")
    p_eval.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "tests" / "fixtures" / "corpus")
    p_eval.add_argument("--out", type=Path, default=PROJECT_ROOT / "evals" / "reports")
    p_eval.add_argument("--judge", action="store_true", help="add an LLM-judge faithfulness score (extra model calls; live only)")
    p_eval.add_argument("--resume", action="store_true", help="save finished questions to evals/reports/.checkpoint-*.jsonl and skip them on rerun (for rate-limited keys)")
    p_eval.add_argument("--retrieval-only", action="store_true", help="measure retrieval and the pre-generation guards only; never call the generation model")
    p_eval.add_argument("--ablation", action="store_true", help="also run unanswerable questions with both guards off")
    return parser


def _run_eval(conn, settings: Settings, args) -> int:
    from ragdemo import evals
    from ragdemo.metrics import load_pricing

    if not args.corpus.exists() or not args.questions.exists():
        print("error: --corpus or --questions path does not exist", file=sys.stderr)
        return 2
    store.migrate(conn, settings.migrations_dir)
    store.check_dimension(conn, settings.embedding_dim)
    embedder = build_embedder(settings)
    generator = build_generator(settings)
    items = evals.load_questions(args.questions)

    print(f"ingesting {args.corpus} with {embedder.model} (idempotent)")
    report = ingest_path(
        conn, embedder, args.corpus,
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, batch_size=settings.embed_batch_size,
        log=print,
    )
    ingest_tokens = report.embed_tokens
    if ingest_tokens is None and report.embedded:
        ingest_tokens = report.embedded * settings.chunk_size // 4  # upper bound; the API reports no counts
    judge = None
    if args.judge:
        if settings.use_fake:
            print("note: --judge is ignored in offline mode", file=sys.stderr)
        else:
            judge = evals.GeminiJudge(generator.client, settings.generation_model, settings.thinking_level,
                                      load_pricing(settings.pricing_path))
    checkpoint = None
    if args.resume:
        tag = "retrieval" if args.retrieval_only else generator.model
        checkpoint = args.out / f".checkpoint-{tag}.jsonl"
    results = evals.run_questions(conn, embedder, generator, settings, items, judge=judge,
                                  ablation=args.ablation, generate=not args.retrieval_only,
                                  checkpoint=checkpoint, progress=print)
    result = evals.build_report(
        results, settings,
        corpus=str(args.corpus.relative_to(PROJECT_ROOT)) if args.corpus.is_relative_to(PROJECT_ROOT) else str(args.corpus),
        corpus_chunks=store.count_chunks(conn, embedder.model),
        ingest_tokens_estimate=ingest_tokens,
        judge_model=judge.model if judge else None,
        ablation=args.ablation and not args.retrieval_only,
        generated=not args.retrieval_only,
        questions_file=str(args.questions.relative_to(PROJECT_ROOT)) if args.questions.is_relative_to(PROJECT_ROOT) else str(args.questions),
    )
    json_path, md_path = evals.write_report(result, args.out)
    if checkpoint and not result["summary"]["errored"] and checkpoint.exists():
        checkpoint.unlink()  # complete: the report supersedes the checkpoint
    s = result["summary"]
    print(
        f"\nrecall@5 hybrid={s['recall_hybrid']['5']} correctness={s['answer_correctness']} errored={s['errored']} "
        f"refusal_accuracy={s['refusal_accuracy']} false_refusals={s['false_refusal_rate']} "
        f"citation_validity={s['citation_validity']}\nreport: {md_path}\n        {json_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    from google.genai import errors as genai_errors

    try:
        return _main(argv)
    except genai_errors.APIError as exc:
        print(f"error: the Gemini API request failed after retries: {type(exc).__name__} {getattr(exc, 'code', '')}: "
              f"{str(exc)[:300]}", file=sys.stderr)
        return 3


def _main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.from_env(fake=True if args.fake else None)

    try:
        conn = store.connect(settings.database_url)
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot connect to the database ({exc}). Is `docker compose up -d` running?", file=sys.stderr)
        return 2

    with conn:
        if args.command == "migrate":
            applied = store.migrate(conn, settings.migrations_dir)
            print("applied: " + ", ".join(applied) if applied else "schema is up to date")
            return 0

        if args.command == "ingest":
            if not args.path.exists():
                print(f"error: {args.path} does not exist", file=sys.stderr)
                return 2
            store.migrate(conn, settings.migrations_dir)
            store.check_dimension(conn, settings.embedding_dim)
            embedder = build_embedder(settings)
            print(f"embedding model: {embedder.model} ({embedder.dim} dims)")
            report = ingest_path(
                conn,
                embedder,
                args.path,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                batch_size=settings.embed_batch_size,
                log=print,
            )
            for warning in report.warnings:
                print(f"warning: {warning}", file=sys.stderr)
            tokens = "n/a" if report.embed_tokens is None else report.embed_tokens
            print(
                f"done: files={report.files} chunks={report.chunks_total} "
                f"already_present={report.already_present} embedded={report.embedded} "
                f"inserted={report.inserted} embed_tokens={tokens}"
            )
            return 0

        if args.command == "eval":
            return _run_eval(conn, settings, args)  # migrates and ingests the eval corpus itself

        if not store.schema_ready(conn):
            print(
                "error: the database has no schema yet. Run `python -m ragdemo migrate` "
                "(or `python -m ragdemo ingest <path>`, which migrates first).",
                file=sys.stderr,
            )
            return 2

        # ask
        store.check_dimension(conn, settings.embedding_dim)
        embedder = build_embedder(settings)
        generator = build_generator(settings)
        result = ask(conn, embedder, generator, settings, args.question, source=args.source, top_k=args.top_k)
        print(result.text)
        if result.chunks:
            print("\nRetrieved:")
            for chunk in result.chunks:
                sim = "-" if chunk.vector_similarity is None else f"{chunk.vector_similarity:.3f}"
                print(f"  [{chunk.label}] id={chunk.id} cosine={sim} rrf={chunk.rrf_score:.4f}")
        m = result.metrics
        print(
            f"\nanswered={m['answered']} top_similarity={m['scores']['top_similarity']} "
            f"total={m['latency_ms']['total']}ms est_cost=${m['est_cost_usd']['total']}"
            f"  (logged to {settings.log_path})"
        )
        return 0
