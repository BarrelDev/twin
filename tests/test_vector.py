from pathlib import Path

import pytest

from twin.ingestion.embedder import Embedder
from twin.storage.vector import SearchResult, VectorStore
from tests.conftest import MockChunk


def _chunk(
    index: int,
    text: str = "Generic filler content.",
    source: str = "/notes/doc.md",
    doc_id: str = "doc",
) -> MockChunk:
    return MockChunk(
        chunk_id=f"{doc_id}_{index}",
        doc_id=doc_id,
        text=text,
        source_path=source,
        heading_path=["Section"],
        chunk_index=index,
        token_count=len(text.split()),
    )


def _zeros() -> list[float]:
    return [0.0] * 768


class TestVectorStoreInit:
    def test_creates_on_fresh_path(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "db")
        assert store is not None

    def test_opens_existing_without_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "db"
        store1 = VectorStore(db_path)
        store1.write_chunks([_chunk(0)], [_zeros()])
        # Must not raise even though the table already exists on disk
        store2 = VectorStore(db_path)
        assert store2 is not None


class TestWriteChunks:
    def test_write_single_chunk_is_searchable(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "db")
        store.write_chunks([_chunk(0)], [_zeros()])
        results = store.search(_zeros(), k=1)
        assert len(results) == 1

    def test_write_100_chunks(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "db")
        store.write_chunks(
            [_chunk(i) for i in range(100)],
            [_zeros() for _ in range(100)],
        )
        results = store.search(_zeros(), k=10)
        assert len(results) == 10

    def test_write_preserves_all_metadata(self, tmp_path: Path) -> None:
        chunk = MockChunk(
            chunk_id="meta_chunk_0",
            doc_id="meta_doc",
            text="Metadata preservation test.",
            source_path="/notes/meta.md",
            heading_path=["Top", "Nested"],
            chunk_index=7,
            token_count=3,
        )
        store = VectorStore(tmp_path / "db")
        store.write_chunks([chunk], [_zeros()])
        result = store.search(_zeros(), k=1)[0]
        assert result.chunk_id == "meta_chunk_0"
        assert result.doc_id == "meta_doc"
        assert result.text == "Metadata preservation test."
        assert result.source_path == "/notes/meta.md"
        assert result.heading_path == ["Top", "Nested"]
        assert result.chunk_index == 7


class TestSearch:
    def test_returns_search_result_instances(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "db")
        store.write_chunks([_chunk(0)], [_zeros()])
        results = store.search(_zeros(), k=1)
        assert isinstance(results[0], SearchResult)

    def test_score_is_float(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "db")
        store.write_chunks([_chunk(0)], [_zeros()])
        result = store.search(_zeros(), k=1)[0]
        assert isinstance(result.score, float)

    def test_heading_path_is_list(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "db")
        store.write_chunks([_chunk(0)], [_zeros()])
        result = store.search(_zeros(), k=1)[0]
        assert isinstance(result.heading_path, list)

    def test_k_limits_result_count(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "db")
        store.write_chunks(
            [_chunk(i) for i in range(20)],
            [_zeros() for _ in range(20)],
        )
        assert len(store.search(_zeros(), k=3)) == 3
        assert len(store.search(_zeros(), k=7)) == 7

    def test_source_path_filter(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "db")
        chunks_a = [_chunk(i, source="/notes/a.md") for i in range(5)]
        chunks_b = [_chunk(i + 5, source="/notes/b.md") for i in range(5)]
        store.write_chunks(chunks_a + chunks_b, [_zeros() for _ in range(10)])
        results = store.search(_zeros(), k=10, source_path="/notes/a.md")
        assert len(results) > 0
        assert all(r.source_path == "/notes/a.md" for r in results)

    def test_source_path_filter_excludes_other_sources(self, tmp_path: Path) -> None:
        store = VectorStore(tmp_path / "db")
        chunks_a = [_chunk(i, source="/notes/a.md") for i in range(5)]
        chunks_b = [_chunk(i + 5, source="/notes/b.md") for i in range(5)]
        store.write_chunks(chunks_a + chunks_b, [_zeros() for _ in range(10)])
        results = store.search(_zeros(), k=10, source_path="/notes/a.md")
        assert not any(r.source_path == "/notes/b.md" for r in results)

    def test_semantic_relevance(self, tmp_path: Path, embedder: Embedder) -> None:
        needle = MockChunk(
            chunk_id="needle_0",
            doc_id="bio",
            text="The mitochondria is the powerhouse of the cell.",
            source_path="/notes/bio.md",
            heading_path=["Biology"],
            chunk_index=49,
            token_count=9,
        )
        filler = [_chunk(i, text=f"Unrelated administrative note number {i}.") for i in range(49)]
        all_chunks = filler + [needle]
        embeddings = embedder.embed_batch([c.text for c in all_chunks])

        store = VectorStore(tmp_path / "db")
        store.write_chunks(all_chunks, embeddings)

        query_vec = embedder.embed_query("What is the powerhouse of the cell?")
        results = store.search(query_vec, k=3)
        assert "needle_0" in [r.chunk_id for r in results]


class TestPersistence:
    def test_data_survives_reconnect(self, tmp_path: Path) -> None:
        db_path = tmp_path / "db"
        chunk = _chunk(0, text="Persistent content check.")

        store1 = VectorStore(db_path)
        store1.write_chunks([chunk], [_zeros()])

        store2 = VectorStore(db_path)
        results = store2.search(_zeros(), k=1)
        assert len(results) == 1
        assert results[0].text == "Persistent content check."

    def test_all_chunks_persist(self, tmp_path: Path) -> None:
        db_path = tmp_path / "db"
        store1 = VectorStore(db_path)
        store1.write_chunks(
            [_chunk(i) for i in range(10)],
            [_zeros() for _ in range(10)],
        )

        store2 = VectorStore(db_path)
        results = store2.search(_zeros(), k=10)
        assert len(results) == 10
