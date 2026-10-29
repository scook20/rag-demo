"""Evaluation harness: a labeled question set run through the real pipeline, scored on retrieval,
refusal, citations and answer correctness, with an optional LLM-judge faithfulness score.

Label file format (evals/questions.yaml), one entry per question:

    - id: q01
      type: lookup | paraphrase | multi-evidence | near-miss | off-topic
      question: "..."
      answerable: true
      evidence:                      # chunks that hold the answer (answerable questions)
        - source: kestrel_reserves_memo.pdf
          page: 1                    # optional
          contains: "2P"             # case-insensitive substring of the chunk text
      key_facts:                     # every fact must appear in the answer; each fact is a list of
        - ["21.7"]                   # acceptable spellings (any one of them matches)
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import psycopg
import yaml

from ragdemo.answer import Answer, ask
from ragdemo.config import Settings
from ragdemo.embedder import Embedder
from ragdemo.generator import Generator
from ragdemo.metrics import dumps_strict, estimate_cost, load_pricing
from ragdemo.models import RetrievedChunk
from ragdemo.retry import with_retries

KS = (1, 3, 5)
THRESHOLD_SWEEP = tuple(round(0.30 + 0.05 * i, 2) for i in range(13))  # 0.30 .. 0.90


# ---- labels -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalItem:
    id: str
    type: str
    question: str
    answerable: bool
    evidence: tuple[dict[str, Any], ...] = ()
    key_facts: tuple[tuple[str, ...], ...] = ()


def load_questions(path: Path) -> list[EvalItem]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items: list[EvalItem] = []
    for entry in raw["questions"]:
        answerable = bool(entry["answerable"])
        evidence = tuple(entry.get("evidence") or ())
        facts = tuple(tuple(str(a) for a in fact) for fact in (entry.get("key_facts") or ()))
        if answerable and not (evidence and facts):
            raise ValueError(f"{entry['id']}: answerable questions need evidence and key_facts")
        if not answerable and (evidence or facts):
            raise ValueError(f"{entry['id']}: unanswerable questions must not list evidence or key_facts")
        items.append(EvalItem(entry["id"], entry["type"], entry["question"], answerable, evidence, facts))
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    return items


# ---- scoring ----------------------------------------------------------------------------------


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def evidence_found(chunks: Sequence[RetrievedChunk], evidence: dict[str, Any]) -> bool:
    """True if any chunk is from the evidence source (and page, when given) and contains its text."""
    needle = _squash(str(evidence["contains"]))
    for c in chunks:
        if c.source != evidence["source"]:
            continue
        if evidence.get("page") is not None and c.page != evidence["page"]:
            continue
        if needle in _squash(c.content):
            return True
    return False


def evidence_recall(chunks: Sequence[RetrievedChunk], evidence: Sequence[dict[str, Any]], k: int) -> float:
    """Fraction of the labeled evidence items present in the first k chunks."""
    top = list(chunks)[:k]
    return sum(evidence_found(top, e) for e in evidence) / len(evidence)


def normalize_answer(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 6,100 -> 6100
    return _squash(text)


def key_fact_hits(answer: str, key_facts: Sequence[Sequence[str]]) -> list[bool]:
    norm = normalize_answer(answer)
    return [any(normalize_answer(alt) in norm for alt in fact) for fact in key_facts]


_CITATION = re.compile(r"\[([^\[\]]+?)\]")


def cites_gold_source(answer: str, evidence: Sequence[dict[str, Any]]) -> bool:
    """True if some citation in the answer points at a labeled evidence source (and page, when given)."""
    for label in (m.strip() for m in _CITATION.findall(answer)):
        for e in evidence:
            if not label.startswith(e["source"]):
                continue
            if e.get("page") is None or f"p.{e['page']}" in label:
                return True
    return False


# ---- optional LLM judge -----------------------------------------------------------------------

JUDGE_PROMPT = """You are grading a retrieval-augmented answer for faithfulness.
Decide whether EVERY factual claim in the answer is supported by the excerpts. Ignore citation markers.
A claim that is not stated in the excerpts is unsupported even if it is true in the real world.
Reply with JSON only: {{"supported": true or false, "unsupported_claims": ["..."]}}

Excerpts:
{excerpts}

Question: {question}

Answer: {answer}"""


@dataclass(frozen=True)
class JudgeResult:
    supported: bool | None  # None when the judge reply could not be parsed
    unsupported_claims: list[str]
    cost_usd: float | None


class Judge(Protocol):
    model: str

    def judge(self, question: str, chunks: Sequence[RetrievedChunk], answer: str) -> JudgeResult: ...


class GeminiJudge:
    """LLM-as-judge faithfulness check. It uses the same model family as the generator, which biases
    the score upward, so treat it as a smoke signal and not a ground truth."""

    def __init__(self, client, model: str, thinking_level: str, pricing: dict[str, Any]):
        self._client, self.model, self._thinking, self._pricing = client, model, thinking_level, pricing

    def judge(self, question: str, chunks: Sequence[RetrievedChunk], answer: str) -> JudgeResult:
        import json

        from google.genai import types

        excerpts = "\n\n".join(f"[{c.label}]\n{c.content}" for c in chunks)
        kwargs: dict[str, Any] = {"response_mime_type": "application/json"}
        if self._thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self._thinking)
        response = with_retries(
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=JUDGE_PROMPT.format(excerpts=excerpts, question=question, answer=answer),
                config=types.GenerateContentConfig(**kwargs),
            )
        )
        usage = response.usage_metadata
        out = (getattr(usage, "candidates_token_count", None) or 0) + (getattr(usage, "thoughts_token_count", None) or 0)
        cost = estimate_cost(self._pricing, self.model, getattr(usage, "prompt_token_count", None), out)
        try:
            data = json.loads(response.text or "")
            return JudgeResult(bool(data["supported"]), [str(c) for c in data.get("unsupported_claims") or []], cost)
        except (ValueError, KeyError, TypeError):
            return JudgeResult(None, [], cost)


# ---- running ----------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _item_result(item: EvalItem, answer: Answer, ks: Sequence[int], generated: bool = True) -> dict[str, Any]:
    m = answer.metrics
    rec: dict[str, Any] = {
        "id": item.id,
        "type": item.type,
        "question": item.question,
        "answerable": item.answerable,
        "answered": answer.answered,
        "refusal_reason": m["refusal_reason"],
        "unsupported_terms": m["unsupported_terms"],
        "top_similarity": m["scores"]["top_similarity"],
        "answer": answer.text,
        "retrieved": [c.label for c in answer.chunks],
        "latency_ms": m["latency_ms"],
        "est_cost_usd": m["est_cost_usd"]["total"],
        "citations": m["citations"],
    }
    if item.answerable:
        retrieval = answer.retrieval
        for name, chunks in (("hybrid", answer.chunks), ("vector", retrieval.vector_ranked), ("fts", retrieval.fts_ranked)):
            rec[f"recall_{name}"] = {str(k): round(evidence_recall(chunks, item.evidence, k), 4) for k in ks}
        if generated:
            hits = key_fact_hits(answer.text, item.key_facts) if answer.answered else [False] * len(item.key_facts)
            rec["key_fact_hits"] = hits
            rec["correct"] = answer.answered and all(hits)
            cited = m["citations"]["cited"]
            rec["citation_valid"] = bool(cited) and m["citations"]["all_in_retrieved"] is True if answer.answered else None
            rec["citation_gold"] = cites_gold_source(answer.text, item.evidence) if answer.answered else None
    return rec


def run_questions(
    conn: psycopg.Connection,
    embedder: Embedder,
    generator: Generator,
    settings: Settings,
    items: Sequence[EvalItem],
    *,
    judge: Judge | None = None,
    ablation: bool = False,
    generate: bool = True,
    checkpoint: Path | None = None,
    ks: Sequence[int] = KS,
    progress: Callable[[str], None] = lambda _m: None,
) -> list[dict[str, Any]]:
    """Run each item through ask() (no request log) and score it. With generate=False the model is never
    called: only retrieval and the two pre-generation guards are measured. An item whose API call still
    fails after retries is recorded with an `error` and left out of the scores. With a checkpoint file,
    finished items are appended to it as they complete, and a rerun skips them (free-tier quotas can
    stop a long run part-way; errored items are not saved, so they are retried)."""
    from google.genai import errors

    signature = {
        "embedding_model": embedder.model, "generation_model": generator.model, "min_similarity": settings.min_similarity,
        "term_guard": settings.term_guard, "top_k": settings.top_k, "generate": generate, "judge": bool(judge),
        "ablation": ablation, "thinking_level": settings.thinking_level,
    }
    done = _load_checkpoint(checkpoint, signature) if checkpoint else {}

    settings = replace(settings, top_k=max(max(ks), settings.top_k))
    if not generate:
        generator = RetrievalOnlyGenerator()
        judge, ablation = None, False
    results: list[dict[str, Any]] = []
    for item in items:
        if item.id in done:
            progress(f"{item.id} (from checkpoint)")
            results.append(done[item.id])
            continue
        try:
            answer = ask(conn, embedder, generator, settings, item.question, log=False)
        except errors.APIError as exc:
            rec = {"id": item.id, "type": item.type, "question": item.question, "answerable": item.answerable,
                   "error": f"{type(exc).__name__} {getattr(exc, 'code', '')}: {str(exc)[:160]}"}
            progress(f"{item.id} ERROR {rec['error']}")
            results.append(rec)
            continue
        rec = _item_result(item, answer, ks, generated=generate)
        try:
            if judge and answer.answered:
                verdict = judge.judge(item.question, answer.chunks, answer.text)
                rec["judge"] = {"supported": verdict.supported, "unsupported_claims": verdict.unsupported_claims, "cost_usd": verdict.cost_usd}
            if ablation and not item.answerable:
                # Same question with both pre-generation guards off: what does the model do alone?
                bare = replace(settings, min_similarity=-1.0, term_guard=False)
                bare_answer = ask(conn, embedder, generator, bare, item.question, log=False)
                rec["model_only_refused"] = not bare_answer.answered
                rec["model_only_answer"] = bare_answer.text
                rec["model_only_cost_usd"] = bare_answer.metrics["est_cost_usd"]["total"]
        except errors.APIError as exc:
            rec["extra_error"] = f"{type(exc).__name__} {getattr(exc, 'code', '')}: {str(exc)[:160]}"
        good = rec.get("correct", not rec["answered"]) if generate or not item.answerable else rec["answered"]
        progress(f"{item.id} {'ok' if good else 'MISS'}" + (f" (extra call failed: {rec['extra_error']})" if "extra_error" in rec else ""))
        if checkpoint and "extra_error" not in rec:
            _append_checkpoint(checkpoint, signature, rec)
        results.append(rec)
    return results


def _load_checkpoint(path: Path, signature: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import json

    if not path.is_file():
        return {}
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0].get("signature") != signature:
        raise ValueError(f"{path} was written with different settings; delete it to start over")
    return {rec["id"]: rec for rec in lines[1:]}


def _append_checkpoint(path: Path, signature: dict[str, Any], rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file()
    with path.open("a", encoding="utf-8") as handle:
        if fresh:
            handle.write(dumps_strict({"signature": signature}) + "\n")
        handle.write(dumps_strict(rec) + "\n")


class RetrievalOnlyGenerator:
    """Stand-in used by --retrieval-only: marks that nothing was generated."""

    model = "none (retrieval-only run)"
    synthetic = True

    def generate(self, question, chunks):
        from ragdemo.generator import GenerationResult

        return GenerationResult("[not generated]", 0, 0, 0)


def threshold_sweep(results: Sequence[dict[str, Any]], thresholds: Sequence[float] = THRESHOLD_SWEEP) -> list[dict[str, Any]]:
    """For each similarity threshold: share of answerable questions that would pass the gate and share
    of unanswerable ones it would block. Uses the recorded best cosine similarity, no extra API calls."""
    results = [r for r in results if "error" not in r]
    answerable = [r["top_similarity"] if r["top_similarity"] is not None else float("-inf") for r in results if r["answerable"]]
    unanswerable = [r["top_similarity"] if r["top_similarity"] is not None else float("-inf") for r in results if not r["answerable"]]
    rows = []
    for t in thresholds:
        rows.append(
            {
                "threshold": t,
                "answerable_passing": round(sum(s >= t for s in answerable) / len(answerable), 4) if answerable else None,
                "unanswerable_blocked": round(sum(s < t for s in unanswerable) / len(unanswerable), 4) if unanswerable else None,
            }
        )
    return rows


def summarize(results: Sequence[dict[str, Any]], ks: Sequence[int] = KS, generated: bool = True) -> dict[str, Any]:
    all_results = results
    results = [r for r in results if "error" not in r]
    ans = [r for r in results if r["answerable"]]
    una = [r for r in results if not r["answerable"]]
    summary: dict[str, Any] = {
        "questions": len(all_results), "scored": len(results), "errored": len(all_results) - len(results),
        "answerable": len(ans), "unanswerable": len(una), "generated": generated,
    }
    for name in ("hybrid", "vector", "fts"):
        summary[f"recall_{name}"] = {str(k): _mean([r[f"recall_{name}"][str(k)] for r in ans]) for k in ks}
    # With generated=False, "answered" only means "got past the two pre-generation guards".
    summary["false_refusal_rate"] = _mean([float(not r["answered"]) for r in ans])
    summary["refusal_accuracy"] = _mean([float(not r["answered"]) for r in una])
    if generated:
        summary["answer_correctness"] = _mean([float(r["correct"]) for r in ans])
        summary["key_fact_recall"] = _mean([sum(r["key_fact_hits"]) / len(r["key_fact_hits"]) for r in ans])
        summary["overall_answer_or_refuse_accuracy"] = _mean([float(r["correct"]) for r in ans] + [float(not r["answered"]) for r in una])
        answered = [r for r in ans if r["answered"]]
        summary["citation_validity"] = _mean([float(r["citation_valid"]) for r in answered])
        summary["citation_gold_source"] = _mean([float(r["citation_gold"]) for r in answered])
    else:
        for key in ("answer_correctness", "key_fact_recall", "overall_answer_or_refuse_accuracy", "citation_validity", "citation_gold_source"):
            summary[key] = None
    reasons: dict[str, int] = {}
    for r in una:
        key = r["refusal_reason"] or "answered"
        reasons[key] = reasons.get(key, 0) + 1
    summary["unanswerable_outcomes"] = reasons
    by_type: dict[str, list[float]] = {}
    for r in ans if generated else []:
        by_type.setdefault(r["type"], []).append(float(r["correct"]))
    summary["correctness_by_type"] = {t: {"n": len(v), "correct": _mean(v)} for t, v in sorted(by_type.items())}
    judged = [r["judge"]["supported"] for r in results if r.get("judge") and r["judge"]["supported"] is not None]
    summary["judge_faithfulness"] = _mean([float(x) for x in judged]) if judged else None
    summary["judge_n"] = len(judged)
    ablated = [r for r in una if "model_only_refused" in r]
    summary["model_only_refusal_accuracy"] = _mean([float(r["model_only_refused"]) for r in ablated]) if ablated else None
    totals = [r["latency_ms"]["total"] for r in results if r["latency_ms"]["total"] is not None]
    summary["latency_ms_total"] = {
        "p50": round(statistics.median(totals), 1) if totals else None,
        "max": round(max(totals), 1) if totals else None,
    }
    sims_a = [r["top_similarity"] for r in ans if r["top_similarity"] is not None]
    sims_u = [r["top_similarity"] for r in una if r["top_similarity"] is not None]
    summary["top_similarity"] = {
        "answerable_min": min(sims_a) if sims_a else None,
        "answerable_median": round(statistics.median(sims_a), 4) if sims_a else None,
        "unanswerable_max": max(sims_u) if sims_u else None,
        "unanswerable_median": round(statistics.median(sims_u), 4) if sims_u else None,
    }
    return summary


def total_cost(results: Sequence[dict[str, Any]]) -> float:
    cost = 0.0
    for r in results:
        cost += r.get("est_cost_usd") or 0.0
        cost += (r.get("judge") or {}).get("cost_usd") or 0.0
        cost += r.get("model_only_cost_usd") or 0.0
    return round(cost, 6)


def build_report(
    results: list[dict[str, Any]],
    settings: Settings,
    *,
    corpus: str,
    corpus_chunks: int,
    ingest_tokens_estimate: int | None,
    judge_model: str | None,
    ablation: bool,
    questions_file: str,
    generated: bool = True,
) -> dict[str, Any]:
    pricing = load_pricing(settings.pricing_path)
    ingest_cost = estimate_cost(pricing, settings.embedding_model, ingest_tokens_estimate) if ingest_tokens_estimate else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "offline-fake" if settings.use_fake else ("live" if generated else "live-retrieval-only"),
        "config": {
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "generation_model": settings.generation_model,
            "thinking_level": settings.thinking_level,
            "min_similarity": settings.min_similarity,
            "term_guard": settings.term_guard,
            "top_k": settings.top_k,
            "candidates": settings.candidates,
            "rrf_k": settings.rrf_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "judge_model": judge_model,
            "ablation": ablation,
            "pricing_verified_on": str(pricing.get("verified_on")),
        },
        "corpus": {"path": corpus, "chunks_in_db_for_model": corpus_chunks},
        "questions_file": questions_file,
        "summary": summarize(results, generated=generated),
        "threshold_sweep": threshold_sweep(results),
        "est_cost_usd": {
            "questions_and_judge": total_cost(results),
            "ingest_embedding_estimate": None if ingest_cost is None else round(ingest_cost, 6),
        },
        "items": results,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    s, c = report["summary"], report["config"]
    lines = [
        f"# RAG eval report ({report['mode']})",
        "",
        f"Generated {report['generated_at']}. Labels: `{report['questions_file']}` "
        f"({s['questions']} questions: {s['answerable']} answerable, {s['unanswerable']} unanswerable"
        + (f", {s['errored']} errored and excluded" if s["errored"] else "")
        + "). "
        f"Corpus: `{report['corpus']['path']}` ({report['corpus']['chunks_in_db_for_model']} chunks).",
        "",
        "Fictional corpus, small builder-authored label set: a regression and sanity check, not a benchmark.",
        "",
        "## Configuration",
        "",
        f"Embedding `{c['embedding_model']}` ({c['embedding_dim']} dims), generation `{c['generation_model']}` "
        f"(thinking `{c['thinking_level'] or 'default'}`), top_k {c['top_k']}, candidates {c['candidates']}, RRF k {c['rrf_k']}, "
        f"min similarity {c['min_similarity']}, term guard {'on' if c['term_guard'] else 'off'}, "
        f"chunks {c['chunk_size']}/{c['chunk_overlap']}.",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k in KS:
        lines.append(f"| Retrieval recall@{k} (hybrid) | {_pct(s['recall_hybrid'][str(k)])} |")
    for k in KS:
        lines.append(f"| Retrieval recall@{k} (vector only) | {_pct(s['recall_vector'][str(k)])} |")
    for k in KS:
        lines.append(f"| Retrieval recall@{k} (keyword only) | {_pct(s['recall_fts'][str(k)])} |")
    if s["generated"]:
        lines += [
            f"| Answer correctness (all key facts present) | {_pct(s['answer_correctness'])} |",
            f"| Key-fact recall | {_pct(s['key_fact_recall'])} |",
            f"| Refusal accuracy on unanswerable | {_pct(s['refusal_accuracy'])} |",
            f"| False refusals on answerable | {_pct(s['false_refusal_rate'])} |",
            f"| Overall answer-or-refuse accuracy | {_pct(s['overall_answer_or_refuse_accuracy'])} |",
            f"| Citation validity (cites only retrieved chunks, at least one) | {_pct(s['citation_validity'])} |",
            f"| Citation cites a gold source/page | {_pct(s['citation_gold_source'])} |",
        ]
    else:
        lines += [
            "| Unanswerable blocked before the model (similarity gate + term guard) | " + _pct(s["refusal_accuracy"]) + " |",
            "| Answerable wrongly blocked before the model | " + _pct(s["false_refusal_rate"]) + " |",
            "| Answer correctness, citations, model refusals | not measured (retrieval-only run, the model was not called) |",
        ]
    if s["judge_faithfulness"] is not None:
        lines.append(f"| LLM-judge faithfulness (judge: `{c['judge_model']}`, n={s['judge_n']}, same model family, biased high) | {_pct(s['judge_faithfulness'])} |")
    if s["model_only_refusal_accuracy"] is not None:
        lines.append(f"| Refusal accuracy with both guards off (model alone) | {_pct(s['model_only_refusal_accuracy'])} |")
    lines += [
        f"| Latency p50 / max per question | {s['latency_ms_total']['p50']} ms / {s['latency_ms_total']['max']} ms |",
        f"| Estimated spend (questions, judge, ablation) | ${report['est_cost_usd']['questions_and_judge']} |",
        "",
        "How unanswerable questions were handled"
        + (": " if s["generated"] else " (`answered` = got past both guards and would reach the model): ")
        + ", ".join(f"{k}: {v}" for k, v in sorted(s["unanswerable_outcomes"].items())),
        "",
        ("Correctness by question type: " + ", ".join(f"{t} {_pct(v['correct'])} (n={v['n']})" for t, v in s["correctness_by_type"].items()) + "\n") if s["generated"] else "",
        f"Top cosine similarity: answerable min {s['top_similarity']['answerable_min']}, median "
        f"{s['top_similarity']['answerable_median']}; unanswerable median {s['top_similarity']['unanswerable_median']}, "
        f"max {s['top_similarity']['unanswerable_max']}.",
        "",
        "## Similarity threshold sweep",
        "",
        "Best-chunk cosine similarity as the only gate (term guard and model refusal not applied).",
        "",
        "| Threshold | Answerable still answered | Unanswerable blocked |",
        "|---|---|---|",
    ]
    for row in report["threshold_sweep"]:
        lines.append(f"| {row['threshold']:.2f} | {_pct(row['answerable_passing'])} | {_pct(row['unanswerable_blocked'])} |")
    lines += ["", "## Per question", "", "| Id | Type | Outcome | Top sim | Recall@5 | Detail |", "|---|---|---|---|---|---|"]
    for r in report["items"]:
        if "error" in r:
            lines.append(f"| {r['id']} | {r['type']} | ERROR (excluded) | - | - | {r['error'].replace('|', '/')} |")
            continue
        if not s["generated"]:
            sim = "-" if r["top_similarity"] is None else f"{r['top_similarity']:.3f}"
            gate = "past guards" if r["answered"] else f"blocked ({r['refusal_reason']})"
            recall = _pct(r["recall_hybrid"]["5"]) if r["answerable"] else "-"
            extra = f" ({', '.join(r['unsupported_terms'])})" if r["unsupported_terms"] else ""
            lines.append(f"| {r['id']} | {r['type']} | {'answerable' if r['answerable'] else 'unanswerable'}: {gate} | {sim} | {recall} | {extra.strip()} |")
            continue
        if r["answerable"]:
            outcome = "correct" if r["correct"] else ("refused" if not r["answered"] else "wrong")
            recall = _pct(r["recall_hybrid"]["5"])
            detail = ""
            if not r["correct"]:
                detail = (r["answer"][:90] + "...") if len(r["answer"]) > 90 else r["answer"]
        else:
            outcome = "refused (good)" if not r["answered"] else "ANSWERED (bad)"
            recall = "-"
            detail = r["refusal_reason"] or (r["answer"][:90] + "...")
            if r["unsupported_terms"]:
                detail += f" ({', '.join(r['unsupported_terms'])})"
        detail = detail.replace("|", "/").replace("\n", " ")
        sim = "-" if r["top_similarity"] is None else f"{r['top_similarity']:.3f}"
        lines.append(f"| {r['id']} | {r['type']} | {outcome} | {sim} | {recall} | {detail} |")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = {"offline-fake": "offline", "live-retrieval-only": "live-retrieval-only"}.get(report["mode"], "live")
    stem = f"eval-{stamp}-{suffix}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(dumps_strict(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path

