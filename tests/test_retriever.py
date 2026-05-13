from pathlib import Path

import pytest

from twin.ingestion.embedder import Embedder
from twin.ingestion.parser import Chunk
from twin.query.retriever import QueryResult, Retriever
from twin.storage.vector import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    chunk_id: str,
    text: str,
    source: str = "/notes/doc.md",
    heading: str = "Section",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("_")[0],
        text=text,
        source_path=source,
        heading_path=[heading],
        chunk_index=0,
        token_count=len(text.split()),
    )


# ---------------------------------------------------------------------------
# Known corpus — 5 signal chunks + 5 unrelated fillers
# ---------------------------------------------------------------------------

_CORPUS: list[tuple[str, str, str]] = [
    # (chunk_id, text, heading)
    (
        "bio_0",
        "The mitochondria is the powerhouse of the cell, generating ATP through oxidative phosphorylation.",
        "Biology",
    ),
    (
        "git_0",
        "git rebase replays commits from one branch on top of another, rewriting history to produce a linear sequence.",
        "Version Control",
    ),
    (
        "python_0",
        "Python uses indentation to delimit code blocks instead of braces, making whitespace syntactically significant.",
        "Programming Languages",
    ),
    (
        "memory_0",
        "RAM is volatile memory: its contents are lost when power is removed, unlike persistent storage such as SSDs.",
        "Computer Hardware",
    ),
    (
        "sql_0",
        "A SQL JOIN combines rows from two or more tables based on a related column, such as a shared foreign key.",
        "Databases",
    ),
    # fillers — semantically distinct from the signal chunks and from each other
    (
        "filler_0",
        "The French Revolution began in 1789 and fundamentally transformed the political landscape of Europe.",
        "History",
    ),
    (
        "filler_1",
        "Photosynthesis converts light energy into chemical energy stored in glucose molecules within plant cells.",
        "Botany",
    ),
    (
        "filler_2",
        "A sonata is a musical composition typically written for one or two instruments in three or four movements.",
        "Music Theory",
    ),
    (
        "filler_3",
        "Supply and demand curves intersect at the equilibrium price where market quantity supplied equals quantity demanded.",
        "Economics",
    ),
    (
        "filler_4",
        "Tectonic plates move a few centimetres per year, driven by convection currents in the Earth's mantle.",
        "Geology",
    ),
]

_QUERIES: list[tuple[str, str]] = [
    # (query text, expected chunk_id in top-3)
    ("What is the powerhouse of the cell?", "bio_0"),
    ("How does git rebase work?", "git_0"),
    ("How does Python define code blocks?", "python_0"),
    ("What happens to RAM when the computer is turned off?", "memory_0"),
    ("How do you combine rows from multiple tables in SQL?", "sql_0"),
]


@pytest.fixture
def populated_store(tmp_path: Path, embedder: Embedder) -> tuple[VectorStore, Embedder]:
    """VectorStore pre-loaded with the known test corpus."""
    chunks = [_chunk(cid, text, heading=heading) for cid, text, heading in _CORPUS]
    embeddings = embedder.embed_batch([c.text for c in chunks])
    store = VectorStore(tmp_path / "db")
    store.write_chunks(chunks, embeddings)
    return store, embedder


# ---------------------------------------------------------------------------
# Unit tests — Retriever behaviour
# ---------------------------------------------------------------------------

class TestRetrieverInit:
    def test_instantiates(self, tmp_path: Path, embedder: Embedder) -> None:
        store = VectorStore(tmp_path / "db")
        retriever = Retriever(store, embedder)
        assert retriever is not None


class TestQuery:
    def test_returns_query_result_instances(
        self, populated_store: tuple[VectorStore, Embedder]
    ) -> None:
        store, embedder = populated_store
        results = Retriever(store, embedder).query("cell biology", k=3)
        assert all(isinstance(r, QueryResult) for r in results)

    def test_k_limits_result_count(
        self, populated_store: tuple[VectorStore, Embedder]
    ) -> None:
        store, embedder = populated_store
        assert len(Retriever(store, embedder).query("test", k=1)) == 1
        assert len(Retriever(store, embedder).query("test", k=3)) == 3

    def test_score_is_float(
        self, populated_store: tuple[VectorStore, Embedder]
    ) -> None:
        store, embedder = populated_store
        result = Retriever(store, embedder).query("cell", k=1)[0]
        assert isinstance(result.score, float)

    def test_heading_path_is_list(
        self, populated_store: tuple[VectorStore, Embedder]
    ) -> None:
        store, embedder = populated_store
        result = Retriever(store, embedder).query("cell", k=1)[0]
        assert isinstance(result.heading_path, list)


# ---------------------------------------------------------------------------
# Retrieval quality — the most important test in Stage 0
# ---------------------------------------------------------------------------

class TestRetrievalQuality:
    def test_correct_chunk_in_top3_for_all_queries(
        self, populated_store: tuple[VectorStore, Embedder]
    ) -> None:
        """
        For each known query, assert the expected chunk appears in the top-3 results.
        All 5 must pass — this is the Stage 0 correctness bar.
        """
        store, embedder = populated_store
        retriever = Retriever(store, embedder)

        failures: list[str] = []
        for query_text, expected_id in _QUERIES:
            results = retriever.query(query_text, k=3)
            returned_ids = [r.chunk_id for r in results]
            if expected_id not in returned_ids:
                failures.append(
                    f"  query={query_text!r}  expected={expected_id}  got={returned_ids}"
                )

        assert not failures, "Retrieval quality failures:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# format_results
# ---------------------------------------------------------------------------

class TestFormatResults:
    def test_returns_rich_table(
        self, populated_store: tuple[VectorStore, Embedder]
    ) -> None:
        from rich.table import Table

        store, embedder = populated_store
        results = Retriever(store, embedder).query("cell", k=3)
        table = Retriever(store, embedder).format_results(results)
        assert isinstance(table, Table)

    def test_table_has_one_row_per_result(
        self, populated_store: tuple[VectorStore, Embedder]
    ) -> None:
        from rich.table import Table

        store, embedder = populated_store
        retriever = Retriever(store, embedder)
        results = retriever.query("cell", k=5)
        table = retriever.format_results(results)
        assert table.row_count == len(results)
