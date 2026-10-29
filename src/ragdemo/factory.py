"""Build the embedder and generator for the current settings (real Gemini or offline fakes)."""

from __future__ import annotations

from ragdemo.config import Settings
from ragdemo.embedder import Embedder, FakeEmbedder, GeminiEmbedder
from ragdemo.generator import FakeGenerator, GeminiGenerator, Generator


def build_embedder(settings: Settings) -> Embedder:
    if settings.use_fake:
        return FakeEmbedder(dim=settings.embedding_dim, model=settings.embedding_model)
    return GeminiEmbedder(settings.embedding_model, settings.embedding_dim, settings.embed_batch_size)


def build_generator(settings: Settings) -> Generator:
    if settings.use_fake:
        return FakeGenerator()
    return GeminiGenerator(settings.generation_model, settings.thinking_level)
