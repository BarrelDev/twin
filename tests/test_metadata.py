from pathlib import Path

import pytest

from twin.storage.metadata import DocRecord, MetadataStore


def _record(
    doc_id: str = "doc_0",
    source_path: str = "/notes/doc.md",
    file_hash: str = "abc123",
    chunk_count: int = 5,
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
) -> DocRecord:
    return DocRecord(
        doc_id=doc_id,
        source_path=source_path,
        file_hash=file_hash,
        ingest_timestamp="2026-05-13T00:00:00",
        chunk_count=chunk_count,
        embedding_model=embedding_model,
    )


class TestMetadataStoreInit:
    def test_creates_on_fresh_path(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        assert store is not None

    def test_opens_existing_without_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "meta.db"
        store1 = MetadataStore(db_path)
        store1.upsert_doc(_record())
        store2 = MetadataStore(db_path)
        assert store2 is not None


class TestGetHash:
    def test_returns_none_for_unknown_path(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        assert store.get_hash("/notes/nonexistent.md") is None

    def test_returns_hash_after_upsert(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record(file_hash="deadbeef"))
        assert store.get_hash("/notes/doc.md") == "deadbeef"

    def test_returns_none_for_different_source_path(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record(source_path="/notes/a.md"))
        assert store.get_hash("/notes/b.md") is None


class TestUpsertDoc:
    def test_record_is_retrievable_after_upsert(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record(doc_id="doc_0", file_hash="aaa", chunk_count=3))
        assert store.get_hash("/notes/doc.md") == "aaa"

    def test_upsert_updates_existing_record(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record(doc_id="doc_0", file_hash="aaa", chunk_count=3))
        store.upsert_doc(_record(doc_id="doc_0", file_hash="bbb", chunk_count=7))
        assert store.get_hash("/notes/doc.md") == "bbb"
        docs = store.list_docs()
        assert len(docs) == 1
        assert docs[0].chunk_count == 7

    def test_hash_match_means_skip(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record(file_hash="stable"))
        stored = store.get_hash("/notes/doc.md")
        # Caller skips re-ingest when hashes match — simulate that logic here
        assert stored == "stable"

    def test_multiple_docs_stored_independently(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record(doc_id="doc_0", source_path="/notes/a.md", file_hash="aaa"))
        store.upsert_doc(_record(doc_id="doc_1", source_path="/notes/b.md", file_hash="bbb"))
        assert store.get_hash("/notes/a.md") == "aaa"
        assert store.get_hash("/notes/b.md") == "bbb"


class TestDeleteDoc:
    def test_delete_removes_record(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record(doc_id="doc_0"))
        store.delete_doc("doc_0")
        assert store.get_hash("/notes/doc.md") is None

    def test_delete_does_not_affect_other_docs(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record(doc_id="doc_0", source_path="/notes/a.md"))
        store.upsert_doc(_record(doc_id="doc_1", source_path="/notes/b.md"))
        store.delete_doc("doc_0")
        assert store.get_hash("/notes/b.md") is not None


class TestListDocs:
    def test_empty_store_returns_empty_list(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        assert store.list_docs() == []

    def test_returns_all_upserted_docs(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        for i in range(5):
            store.upsert_doc(_record(doc_id=f"doc_{i}", source_path=f"/notes/{i}.md"))
        assert len(store.list_docs()) == 5

    def test_returns_doc_record_instances(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record())
        docs = store.list_docs()
        assert isinstance(docs[0], DocRecord)

    def test_all_fields_preserved(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.upsert_doc(_record(
            doc_id="doc_0",
            source_path="/notes/doc.md",
            file_hash="abc123",
            chunk_count=5,
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
        ))
        doc = store.list_docs()[0]
        assert doc.doc_id == "doc_0"
        assert doc.source_path == "/notes/doc.md"
        assert doc.file_hash == "abc123"
        assert doc.chunk_count == 5
        assert doc.embedding_model == "nomic-ai/nomic-embed-text-v1.5"


class TestPersistence:
    def test_data_survives_reconnect(self, tmp_path: Path) -> None:
        db_path = tmp_path / "meta.db"
        MetadataStore(db_path).upsert_doc(_record(file_hash="persisted"))
        assert MetadataStore(db_path).get_hash("/notes/doc.md") == "persisted"

    def test_all_docs_survive_reconnect(self, tmp_path: Path) -> None:
        db_path = tmp_path / "meta.db"
        store1 = MetadataStore(db_path)
        for i in range(10):
            store1.upsert_doc(_record(doc_id=f"doc_{i}", source_path=f"/notes/{i}.md"))
        assert len(MetadataStore(db_path).list_docs()) == 10
