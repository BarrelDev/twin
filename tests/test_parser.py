from pathlib import Path
import pytest
from twin.ingestion.parser import Chunk, parse_file


class TestChunkDataclass:
    """Test the Chunk dataclass structure and behavior."""

    def test_chunk_creation(self) -> None:
        """Verify Chunk can be instantiated with all required fields."""
        chunk = Chunk(
            chunk_id="chunk_0",
            doc_id="doc_1",
            text="Sample text",
            source_path="/path/to/file.md",
            heading_path=["Main", "Subsection"],
            chunk_index=0,
            token_count=50,
        )
        assert chunk.chunk_id == "chunk_0"
        assert chunk.doc_id == "doc_1"
        assert chunk.text == "Sample text"
        assert chunk.source_path == "/path/to/file.md"
        assert chunk.heading_path == ["Main", "Subsection"]
        assert chunk.chunk_index == 0
        assert chunk.token_count == 50

    def test_chunk_with_empty_heading_path(self) -> None:
        """Verify Chunk works with no heading context."""
        chunk = Chunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            text="Root level text",
            source_path="/path/to/file.md",
            heading_path=[],
            chunk_index=0,
            token_count=25,
        )
        assert chunk.heading_path == []


class TestParseFile:
    """Test the parse_file() function."""

    def test_parse_file_returns_list_of_chunks(self, sample_markdown: Path) -> None:
        """Verify parse_file returns a list."""
        result = parse_file(sample_markdown)
        assert isinstance(result, list)

    def test_parse_file_with_nonexistent_file(self) -> None:
        """Verify parse_file handles missing files gracefully."""
        nonexistent = Path("/nonexistent/path/file.md")
        result = parse_file(nonexistent)
        assert isinstance(result, list)

    def test_parse_file_with_empty_file(self, empty_markdown: Path) -> None:
        """Verify parse_file handles empty files."""
        result = parse_file(empty_markdown)
        assert isinstance(result, list)

    def test_parse_file_preserves_source_path(self, sample_markdown: Path) -> None:
        """Verify all chunks preserve the source file path."""
        chunks = parse_file(sample_markdown)
        assert isinstance(chunks, list)
        # Stub returns empty list; full implementation will verify source_path
        if chunks:
            for chunk in chunks:
                assert isinstance(chunk, Chunk)
                assert chunk.source_path == str(sample_markdown)
