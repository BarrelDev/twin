"""Tests for context formatting and source extraction."""

import pytest

from twin.query.retriever import QueryResult
from twin.rag.context import (
    FormattedContext,
    extract_sources,
    format_chunks_as_context,
    prepare_rag_context,
)


@pytest.fixture
def sample_chunks() -> list[QueryResult]:
    """Sample QueryResult chunks for testing context formatting."""
    return [
        QueryResult(
            chunk_id="chunk_001",
            text="This is the first chunk about machine learning.",
            source_path="/home/user/notes/ai/ml.md",
            heading_path=["Machine Learning", "Basics"],
            score=0.95,
        ),
        QueryResult(
            chunk_id="chunk_002",
            text="This is the second chunk about neural networks.",
            source_path="/home/user/notes/ai/nn.md",
            heading_path=["Neural Networks", "Architecture"],
            score=0.87,
        ),
        QueryResult(
            chunk_id="chunk_003",
            text="This is a third chunk also about machine learning.",
            source_path="/home/user/notes/ai/ml.md",
            heading_path=["Machine Learning", "Advanced"],
            score=0.82,
        ),
    ]


@pytest.fixture
def duplicate_source_chunks() -> list[QueryResult]:
    """Chunks with duplicate sources (same file and heading path)."""
    return [
        QueryResult(
            chunk_id="chunk_001",
            text="First mention of this topic.",
            source_path="/home/user/notes/doc.md",
            heading_path=["Section A"],
            score=0.9,
        ),
        QueryResult(
            chunk_id="chunk_002",
            text="Second mention of the same topic.",
            source_path="/home/user/notes/doc.md",
            heading_path=["Section A"],
            score=0.85,
        ),
    ]


@pytest.fixture
def empty_heading_path_chunks() -> list[QueryResult]:
    """Chunks with empty heading paths."""
    return [
        QueryResult(
            chunk_id="chunk_001",
            text="Chunk at document root.",
            source_path="/home/user/notes/doc.md",
            heading_path=[],
            score=0.9,
        ),
    ]


class TestFormatChunksAsContext:
    """Tests for format_chunks_as_context()."""

    def test_formats_single_chunk_with_source(self) -> None:
        """Verify a single chunk is formatted with source attribution."""
        chunks = [
            QueryResult(
                chunk_id="chunk_001",
                text="Sample text",
                source_path="/path/to/notes.md",
                heading_path=["Section"],
                score=0.9,
            ),
        ]
        result = format_chunks_as_context(chunks)

        assert "[source:" in result
        assert "notes.md" in result
        assert "Section" in result
        assert "Sample text" in result

    def test_formats_multiple_chunks_with_attribution(
        self, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify multiple chunks are formatted with source info on each."""
        result = format_chunks_as_context(sample_chunks)

        # Each file should appear in the output
        assert "ml.md" in result
        assert "nn.md" in result

        # Each text snippet should appear
        assert "machine learning" in result
        assert "neural networks" in result

        # Source markers should be present
        assert "[source:" in result

    def test_includes_heading_path_in_source(
        self, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify heading path is included in source attribution."""
        result = format_chunks_as_context(sample_chunks)

        # All heading path components should appear
        assert "Machine Learning" in result
        assert "Basics" in result
        assert "Advanced" in result
        assert "Neural Networks" in result
        assert "Architecture" in result

    def test_empty_chunks_list_returns_empty_string(self) -> None:
        """Verify empty chunks list produces empty context string."""
        result = format_chunks_as_context([])
        assert result == ""

    def test_handles_empty_heading_path(
        self, empty_heading_path_chunks: list[QueryResult]
    ) -> None:
        """Verify chunks with empty heading paths are handled."""
        result = format_chunks_as_context(empty_heading_path_chunks)

        # Should still include source and text
        assert "[source:" in result
        assert "doc.md" in result
        assert "Chunk at document root" in result

    def test_preserves_chunk_text_content(self) -> None:
        """Verify text content is preserved exactly as provided."""
        text_content = "Important\nMulti-line\nText content"
        chunks = [
            QueryResult(
                chunk_id="chunk_001",
                text=text_content,
                source_path="/test.md",
                heading_path=["Test"],
                score=0.9,
            ),
        ]
        result = format_chunks_as_context(chunks)

        # The exact text should be preserved
        assert text_content in result


class TestExtractSources:
    """Tests for extract_sources()."""

    def test_deduplicates_sources(
        self, duplicate_source_chunks: list[QueryResult]
    ) -> None:
        """Verify sources are deduplicated by path + heading_path."""
        sources = extract_sources(duplicate_source_chunks)

        # Should only have one source even though there are two chunks
        assert len(sources) == 1
        assert sources[0]["path"] == "doc.md"
        assert sources[0]["heading_path"] == ["Section A"]

    def test_preserves_order_of_first_appearance(
        self, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify deduplication preserves order of first appearance."""
        sources = extract_sources(sample_chunks)

        # ml.md appears first (chunk_001), then nn.md (chunk_002), then ml.md again
        # Should have: ml.md, nn.md, ml.md (from different heading path)
        assert len(sources) == 3
        assert sources[0]["path"] == "ml.md"
        assert sources[0]["heading_path"] == ["Machine Learning", "Basics"]
        assert sources[1]["path"] == "nn.md"
        assert sources[2]["path"] == "ml.md"
        assert sources[2]["heading_path"] == ["Machine Learning", "Advanced"]

    def test_extracts_filename_from_path(self) -> None:
        """Verify only the filename is extracted, not the full path."""
        chunks = [
            QueryResult(
                chunk_id="chunk_001",
                text="Test",
                source_path="/deeply/nested/path/to/document.md",
                heading_path=["Heading"],
                score=0.9,
            ),
        ]
        sources = extract_sources(chunks)

        assert sources[0]["path"] == "document.md"
        # Full path should NOT be in the result
        assert "deeply" not in sources[0]["path"]
        assert "/" not in sources[0]["path"]

    def test_empty_chunks_returns_empty_list(self) -> None:
        """Verify empty chunks list produces empty sources list."""
        sources = extract_sources([])
        assert sources == []

    def test_handles_empty_heading_path(
        self, empty_heading_path_chunks: list[QueryResult]
    ) -> None:
        """Verify chunks with empty heading paths are handled."""
        sources = extract_sources(empty_heading_path_chunks)

        assert len(sources) == 1
        assert sources[0]["path"] == "doc.md"
        assert sources[0]["heading_path"] == []

    def test_source_dict_structure(self) -> None:
        """Verify returned source dicts have the correct keys."""
        chunks = [
            QueryResult(
                chunk_id="chunk_001",
                text="Test",
                source_path="/test.md",
                heading_path=["Heading"],
                score=0.9,
            ),
        ]
        sources = extract_sources(chunks)

        assert len(sources) == 1
        assert "path" in sources[0]
        assert "heading_path" in sources[0]
        assert isinstance(sources[0]["heading_path"], list)


class TestPrepareRagContext:
    """Tests for prepare_rag_context()."""

    def test_returns_formatted_context_object(
        self, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify the function returns a FormattedContext with text and sources."""
        result = prepare_rag_context(sample_chunks)

        assert isinstance(result, FormattedContext)
        assert isinstance(result.text, str)
        assert isinstance(result.sources, list)

    def test_text_and_sources_are_consistent(
        self, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify sources match the chunks present in the formatted text."""
        result = prepare_rag_context(sample_chunks)

        # All source files should appear in the text
        for source in result.sources:
            assert source["path"] in result.text

        # All source headings should appear in the text
        for source in result.sources:
            for heading in source["heading_path"]:
                assert heading in result.text

    def test_empty_chunks_returns_empty_context(self) -> None:
        """Verify empty chunks list returns FormattedContext with empty fields."""
        result = prepare_rag_context([])

        assert isinstance(result, FormattedContext)
        assert result.text == ""
        assert result.sources == []

    def test_sources_are_deduplicated_in_result(
        self, duplicate_source_chunks: list[QueryResult]
    ) -> None:
        """Verify sources in the result are deduplicated."""
        result = prepare_rag_context(duplicate_source_chunks)

        # Both chunks have the same source, so only one should appear
        assert len(result.sources) == 1

    def test_formatted_context_preserves_chunk_order(
        self, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify chunks appear in the formatted text in order."""
        result = prepare_rag_context(sample_chunks)

        # Find positions of each chunk's text
        ml_pos = result.text.find("machine learning")
        nn_pos = result.text.find("neural networks")
        advanced_pos = result.text.find("Advanced")

        # They should appear in order (ml, nn, then advanced under ml)
        assert ml_pos < nn_pos
        assert nn_pos < advanced_pos
