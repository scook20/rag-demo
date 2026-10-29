"""Embedding interface with a Gemini implementation and a deterministic offline fake."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from ragdemo.retry import with_retries


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    tokens: int | None  # None when the API does not report token counts


class Embedder(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> EmbedResult: ...

    def embed_query(self, text: str) -> EmbedResult: ...


def l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else list(vec)


class GeminiEmbedder:
    """google-genai embed_content. Works against the Gemini Developer API (API key) now and
    Vertex AI later: genai.Client() picks the backend from environment variables."""

    def __init__(self, model: str, dim: int, batch_size: int = 100, client=None):
        if model.startswith("gemini-embedding-2"):
            raise ValueError(
                "gemini-embedding-2 is not supported in v1: it has no task_type and merges a list of "
                "inputs into one vector. Use gemini-embedding-001 (see docs/decisions.md)."
            )
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        if client is None:
            from google import genai

            client = genai.Client()
        self._client = client

    def _embed(self, texts: Sequence[str], task_type: str) -> EmbedResult:
        from google.genai import types

        vectors: list[list[float]] = []
        total_tokens = 0
        saw_tokens = False
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            response = with_retries(
                lambda b=batch: self._client.models.embed_content(
                    model=self.model,
                    contents=b,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
                )
            )
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise RuntimeError(f"expected {len(batch)} embeddings, got {len(embeddings)}")
            for emb in embeddings:
                # Truncated dimensions are not unit length for gemini-embedding-001: normalize.
                vectors.append(l2_normalize(emb.values))
                stats = getattr(emb, "statistics", None)
                count = getattr(stats, "token_count", None) if stats else None
                if count is not None:
                    saw_tokens = True
                    total_tokens += int(count)
        return EmbedResult(vectors=vectors, tokens=total_tokens if saw_tokens else None)

    def embed_documents(self, texts: Sequence[str]) -> EmbedResult:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> EmbedResult:
        return self._embed([text], "RETRIEVAL_QUERY")


_STOPWORDS = frozenset(
    "a an and are as at be been by do does for from has have how in is it its of on or that the "
    "this to was were what when where which who whom why will with about into than then there "
    "these those their they you your can could should would not no yes i we our".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def fake_tokens(text: str) -> list[str]:
    tokens = []
    for word in _WORD.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]  # crude plural stem so "reserves" matches "reserve"
        tokens.append(word)
    return tokens


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors. No network, no key. Texts that share
    words get high cosine similarity, which is enough to test retrieval end to end."""

    def __init__(self, dim: int = 768, model: str = "fake-hash-embedding"):
        self.model = model
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in fake_tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        return l2_normalize(vec)

    def embed_documents(self, texts: Sequence[str]) -> EmbedResult:
        return EmbedResult([self._vector(t) for t in texts], sum(len(t.split()) for t in texts))

    def embed_query(self, text: str) -> EmbedResult:
        return EmbedResult([self._vector(text)], len(text.split()))
