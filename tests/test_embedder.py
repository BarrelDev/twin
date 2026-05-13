import math
from pathlib import Path

import pytest

from twin.config import AppConfig, EmbeddingModel
from twin.ingestion.embedder import Embedder, build_embedder


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Session-scoped Embedder so the model loads only once."""
    return Embedder(model_name=EmbeddingModel.NOMIC.value, dim=768)


class TestEmbedBatch:
    """Tests for embed_batch()."""

    def test_returns_correct_count(self, embedder: Embedder) -> None:
        """Verify one vector is returned per input text."""
        texts = [f"Sample sentence number {i}." for i in range(10)]
        result = embedder.embed_batch(texts)
        assert len(result) == 10

    def test_returns_correct_dimensions(self, embedder: Embedder) -> None:
        """Verify each embedding has the expected 768 dimensions."""
        result = embedder.embed_batch(["Hello world.", "Testing embeddings."])
        for vec in result:
            assert len(vec) == 768

    def test_returns_lists_of_floats(self, embedder: Embedder) -> None:
        """Verify output type is list[list[float]]."""
        result = embedder.embed_batch(["Test text."])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_empty_input_returns_empty(self, embedder: Embedder) -> None:
        """Verify empty input produces empty output."""
        assert embedder.embed_batch([]) == []

    def test_preserves_order(self, embedder: Embedder) -> None:
        """Verify output ordering matches input ordering."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        result = embedder.embed_batch(texts)
        # Re-embed individually and compare
        for i, text in enumerate(texts):
            single = embedder.embed_batch([text])[0]
            assert result[i] == pytest.approx(single, abs=1e-5)


class TestEmbedQuery:
    """Tests for embed_query()."""

    def test_returns_correct_dimensions(self, embedder: Embedder) -> None:
        """Verify query embedding has 768 dimensions."""
        result = embedder.embed_query("What is machine learning?")
        assert len(result) == 768

    def test_returns_list_of_floats(self, embedder: Embedder) -> None:
        """Verify output is list[float]."""
        result = embedder.embed_query("Test query.")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_semantic_similarity(self, embedder: Embedder) -> None:
        """Verify semantically similar strings score higher than unrelated ones."""
        similar_a = "The cat sat on the mat."
        similar_b = "A cat rested on a rug."
        unrelated = "Quantum mechanics describes subatomic particles."

        vec_a = embedder.embed_query(similar_a)
        vec_b = embedder.embed_query(similar_b)
        vec_c = embedder.embed_query(unrelated)

        sim_similar = _cosine_similarity(vec_a, vec_b)
        sim_unrelated = _cosine_similarity(vec_a, vec_c)

        assert sim_similar > sim_unrelated


class TestBuildEmbedder:
    """Tests for the build_embedder() factory."""

    def test_build_from_config(self, tmp_path: Path) -> None:
        """Verify build_embedder reads model name and dim from config."""
        config = AppConfig(
            data_dir=tmp_path,
            embed_model=EmbeddingModel.NOMIC,
            chunk_tokens=512,
            overlap_tokens=64,
            top_k=5,
        )
        emb = build_embedder(config)
        assert emb.model_name == EmbeddingModel.NOMIC.value
        assert emb.dim == 768
