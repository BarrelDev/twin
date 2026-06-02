import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from twin.config import EmbeddingModel
from twin.ingestion.embedder import Embedder


@dataclass
class MockChunk:
    """Mock Chunk — mirrors the Rust Chunk structure for testing."""
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    heading_path: list[str]
    chunk_index: int
    token_count: int


# ── Real fixtures (session-scoped for performance) ───────────────────────────

@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Session-scoped Embedder — loads the model once for the entire test run."""
    return Embedder(model_name=EmbeddingModel.NOMIC.value, dim=768)


# ── Mock fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    """Mock Embedder returning deterministic fixed embeddings without loading a model."""

    class _MockEmbedder:
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 768 for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 768

    return _MockEmbedder()


@pytest.fixture
def mock_llm_provider():
    """Mock LLMProvider returning canned responses and recording all calls made."""
    from twin.llm.base import LLMProvider

    class _MockResponse:
        """Minimal response object that satisfies extract_answer()."""
        def __init__(self, text: str) -> None:
            self.content = [type("Block", (), {"text": text, "type": "text"})()]
            self.stop_reason = "end_turn"
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 5})()

    class _MockLLMProvider(LLMProvider):
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.response_text = "Mock response"
            self.stream_chunks: list[str] = ["Mock ", "response"]

        def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict] | None = None,
            system: str = "",
        ) -> Any:
            self.calls.append({"method": "complete", "messages": messages, "tools": tools})
            return _MockResponse(self.response_text)

        def extract_answer(self, response: Any) -> str:
            return response.content[0].text

        def list_models(self) -> list[str]:
            return ["mock-model"]

        async def stream(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict] | None = None,
            system: str = "",
        ) -> AsyncIterator[str]:
            self.calls.append({"method": "stream", "messages": messages})
            for chunk in self.stream_chunks:
                yield chunk

        def estimate_cost(
            self, prompt_tokens: int, completion_tokens: int
        ) -> float | None:
            return 0.0001

    return _MockLLMProvider()


# ── Storage fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def tmp_lance_db(tmp_path: Path):
    """Temporary LanceDB vector store, cleaned up after the test."""
    from twin.storage.vector import VectorStore
    return VectorStore(tmp_path / "lancedb")


@pytest.fixture
def tmp_sqlite(tmp_path: Path):
    """Temporary SQLite metadata store, cleaned up after the test."""
    from twin.storage.metadata import MetadataStore
    return MetadataStore(tmp_path / "meta.db")


# ── Vault fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Temporary Obsidian vault with sample notes and an Agents/ directory."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Agents").mkdir()

    (vault / "note1.md").write_text(
        "---\ntitle: Note One\ntags: [test, project]\n---\n"
        "# Note One\n\nContent of note one.\n\n"
        "## Section\n\nSection content with [[Note Two]] link.\n"
    )
    (vault / "note2.md").write_text(
        "---\ntitle: Note Two\n---\n"
        "# Note Two\n\nContent with #tag/nested reference to [[Note One|One]].\n"
    )

    return vault


# ── Markdown corpus fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    """A sample Markdown file for testing."""
    md_file = tmp_path / "test_doc.md"
    md_file.write_text(
        """# Main Heading

This is an introduction paragraph.
It has multiple sentences.

## Subsection

Another paragraph with some content.
More text here.

### Deeper Section

Final paragraph for testing.
"""
    )
    return md_file


@pytest.fixture
def multifile_corpus(tmp_path: Path) -> Path:
    """A corpus of multiple Markdown files for testing."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    (notes_dir / "note1.md").write_text(
        """# First Note

This is the first note.

## Section A
Content of section A.

## Section B
Content of section B.
"""
    )

    (notes_dir / "note2.md").write_text(
        """# Second Note

Different content here.

## Analysis
Some analysis content.
"""
    )

    return notes_dir


@pytest.fixture
def empty_markdown(tmp_path: Path) -> Path:
    """An empty Markdown file for edge case testing."""
    md_file = tmp_path / "empty.md"
    md_file.write_text("")
    return md_file


@pytest.fixture
def markdown_no_headings(tmp_path: Path) -> Path:
    """A Markdown file with only paragraphs, no headings."""
    md_file = tmp_path / "no_headings.md"
    md_file.write_text(
        """Just plain text here.

No headings at all.

Just paragraphs separated by blank lines.
"""
    )
    return md_file
