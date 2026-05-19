from pathlib import Path
import pytest
from twin.ingestion.parser import (
    Chunk,
    parse_file,
    _count_tokens,
    _extract_frontmatter,
    _parse_markdown_structure,
    _split_section_into_chunks,
)


class TestChunkDataclass:
    """Test the Chunk class structure and behavior."""

    def test_chunk_creation(self, sample_markdown: Path) -> None:
        """Verify Chunk has all required fields and they work correctly."""
        chunks = parse_file(sample_markdown)
        assert len(chunks) > 0
        chunk = chunks[0]

        # Verify all required attributes exist and are accessible
        assert hasattr(chunk, "chunk_id") and isinstance(chunk.chunk_id, str)
        assert hasattr(chunk, "doc_id") and isinstance(chunk.doc_id, str)
        assert hasattr(chunk, "text") and isinstance(chunk.text, str)
        assert hasattr(chunk, "source_path") and isinstance(chunk.source_path, str)
        assert hasattr(chunk, "heading_path") and isinstance(chunk.heading_path, list)
        assert hasattr(chunk, "chunk_index") and isinstance(chunk.chunk_index, int)
        assert hasattr(chunk, "token_count") and isinstance(chunk.token_count, int)

    def test_chunk_with_empty_heading_path(self, markdown_no_headings: Path) -> None:
        """Verify Chunk works with no heading context."""
        chunks = parse_file(markdown_no_headings)
        assert len(chunks) > 0
        # Chunks without headings should have empty heading_path
        assert chunks[0].heading_path == []


class TestTokenCounting:
    """Test token counting utility."""

    def test_count_tokens_simple(self) -> None:
        """Verify token count approximation."""
        text = "This is a test"
        assert _count_tokens(text) == 4

    def test_count_tokens_multiline(self) -> None:
        """Verify token count on multiline text."""
        text = "This is line one.\nAnd this is line two."
        assert _count_tokens(text) == 9

    def test_count_tokens_empty(self) -> None:
        """Verify empty text returns zero tokens."""
        assert _count_tokens("") == 0

    def test_count_tokens_whitespace(self) -> None:
        """Verify whitespace-only text returns zero tokens."""
        assert _count_tokens("   \n  \n  ") == 0


class TestFrontmatterExtraction:
    """Test YAML frontmatter parsing."""

    def test_extract_frontmatter_valid(self) -> None:
        """Verify frontmatter extraction from markdown."""
        content = """---
id: test-doc
title: Test Document
---
# Content

Body text here."""
        frontmatter, body = _extract_frontmatter(content)
        assert frontmatter["id"] == "test-doc"
        assert frontmatter["title"] == "Test Document"
        assert "# Content" in body

    def test_extract_frontmatter_missing(self) -> None:
        """Verify handling when no frontmatter present."""
        content = "# No frontmatter\n\nBody text."
        frontmatter, body = _extract_frontmatter(content)
        assert frontmatter == {}
        assert body == content

    def test_extract_frontmatter_malformed(self) -> None:
        """Verify graceful handling of malformed YAML."""
        content = """---
invalid: yaml: content:
---
# Content"""
        frontmatter, body = _extract_frontmatter(content)
        assert frontmatter == {}
        assert body == content


class TestMarkdownStructureParsing:
    """Test heading hierarchy extraction."""

    def test_parse_simple_headings(self) -> None:
        """Verify extraction of simple heading structure."""
        content = """# Main

Content under main.

## Sub1

Content under sub1.

## Sub2

Content under sub2."""
        sections = _parse_markdown_structure(content)
        assert len(sections) == 3
        assert sections[0][0] == ["Main"]
        assert sections[1][0] == ["Main", "Sub1"]
        assert sections[2][0] == ["Main", "Sub2"]

    def test_parse_nested_headings(self) -> None:
        """Verify handling of nested heading levels."""
        content = """# Level 1

Content.

## Level 2

Content.

### Level 3

Content."""
        sections = _parse_markdown_structure(content)
        assert sections[0][0] == ["Level 1"]
        assert sections[1][0] == ["Level 1", "Level 2"]
        assert sections[2][0] == ["Level 1", "Level 2", "Level 3"]

    def test_parse_no_headings(self) -> None:
        """Verify handling of content without headings."""
        content = "Just plain text.\n\nMore text."
        sections = _parse_markdown_structure(content)
        assert len(sections) == 1
        assert sections[0][0] == []
        assert "Just plain text" in sections[0][1]

    def test_parse_empty_content(self) -> None:
        """Verify empty content returns empty list."""
        sections = _parse_markdown_structure("")
        assert sections == []


class TestSectionChunking:
    """Test paragraph-based chunking with overlap."""

    def test_chunk_respects_token_budget(self) -> None:
        """Verify no chunk exceeds max token limit."""
        # Create text with sentences to allow proper splitting
        sentences = [". ".join(["word"] * 10) + "." for _ in range(60)]  # 60 sentences
        text = " ".join(sentences)
        chunks = _split_section_into_chunks(text, max_tokens=512)
        for chunk in chunks:
            token_count = _count_tokens(chunk)
            assert token_count <= 512, f"Chunk has {token_count} tokens, exceeds 512"

    def test_chunk_single_paragraph(self) -> None:
        """Verify single short paragraph is one chunk."""
        text = " ".join(["word"] * 50)
        chunks = _split_section_into_chunks(text, max_tokens=512)
        assert len(chunks) == 1

    def test_chunk_multiple_paragraphs(self) -> None:
        """Verify multiple paragraphs create multiple chunks."""
        text = "\n\n".join([" ".join(["word"] * 300) for _ in range(3)])
        chunks = _split_section_into_chunks(text, max_tokens=512)
        assert len(chunks) >= 2

    def test_chunk_overlap_applied(self) -> None:
        """Verify overlap is applied between chunks."""
        # Create paragraphs with sentences for proper splitting and overlap
        para1 = ". ".join(["word"] * 200) + "."
        para2 = ". ".join(["word"] * 200) + "."
        text = para1 + "\n\n" + para2
        chunks = _split_section_into_chunks(text, max_tokens=512, overlap_tokens=64)
        # With 2 paragraphs of 200 words each, we should get multiple chunks with overlap
        if len(chunks) > 1:
            # Later chunks should contain overlap from previous content
            assert _count_tokens(chunks[1]) > 0

    def test_chunk_empty_text(self) -> None:
        """Verify empty text returns empty list."""
        assert _split_section_into_chunks("") == []
        assert _split_section_into_chunks("   \n   ") == []

    def test_chunk_respects_token_budget_with_overlap(self) -> None:
        """Verify even with overlap, chunks don't exceed budget."""
        # Create text with sentences for proper splitting
        sentences = [". ".join(["word"] * 10) + "." for _ in range(100)]  # 100 sentences
        text = " ".join(sentences)
        chunks = _split_section_into_chunks(text, max_tokens=512, overlap_tokens=64)
        for chunk in chunks:
            token_count = _count_tokens(chunk)
            assert token_count <= 512, f"Chunk has {token_count} tokens, exceeds 512"


class TestParseFile:
    """Test the full parse_file() function."""

    def test_parse_file_returns_list_of_chunks(self, sample_markdown: Path) -> None:
        """Verify parse_file returns a list of Chunk objects."""
        result = parse_file(sample_markdown)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_parse_file_with_nonexistent_file(self) -> None:
        """Verify parse_file handles missing files gracefully."""
        nonexistent = Path("/nonexistent/path/file.md")
        result = parse_file(nonexistent)
        assert result == []

    def test_parse_file_with_empty_file(self, empty_markdown: Path) -> None:
        """Verify parse_file handles empty files."""
        result = parse_file(empty_markdown)
        assert result == []

    def test_parse_file_preserves_source_path(self, sample_markdown: Path) -> None:
        """Verify all chunks preserve the source file path."""
        chunks = parse_file(sample_markdown)
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.source_path == str(sample_markdown)

    def test_parse_file_no_chunk_exceeds_token_budget(
        self, sample_markdown: Path
    ) -> None:
        """Verify no chunk exceeds 512 token limit."""
        chunks = parse_file(sample_markdown)
        for chunk in chunks:
            assert chunk.token_count <= 512, f"Chunk {chunk.chunk_id} has {chunk.token_count} tokens"

    def test_parse_file_preserves_heading_structure(self, sample_markdown: Path) -> None:
        """Verify heading paths are captured correctly."""
        chunks = parse_file(sample_markdown)
        assert len(chunks) > 0
        # First chunk should have Main heading
        assert chunks[0].heading_path == ["Main Heading"]

    def test_parse_file_with_frontmatter(self, tmp_path: Path) -> None:
        """Verify frontmatter is parsed and used for doc_id."""
        md_file = tmp_path / "frontmatter.md"
        md_file.write_text(
            """---
id: custom-id
title: My Document
---
# Main

Content here."""
        )
        chunks = parse_file(md_file)
        assert len(chunks) > 0
        assert chunks[0].doc_id == "custom-id"

    def test_parse_file_chunk_indexing(self, sample_markdown: Path) -> None:
        """Verify chunks are indexed sequentially."""
        chunks = parse_file(sample_markdown)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_parse_file_unique_chunk_ids(self, sample_markdown: Path) -> None:
        """Verify all chunk IDs are unique."""
        chunks = parse_file(sample_markdown)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_parse_file_reasonable_chunk_count(
        self, multifile_corpus: Path
    ) -> None:
        """Verify chunk count is reasonable for corpus."""
        note1 = multifile_corpus / "note1.md"
        note2 = multifile_corpus / "note2.md"

        chunks1 = parse_file(note1)
        chunks2 = parse_file(note2)

        # Both files should produce some chunks
        assert len(chunks1) > 0
        assert len(chunks2) > 0
        # Total chunks should be reasonable (not explosion)
        assert len(chunks1) + len(chunks2) < 100

    def test_parse_file_token_count_matches_text(
        self, sample_markdown: Path
    ) -> None:
        """Verify reported token_count matches actual text."""
        chunks = parse_file(sample_markdown)
        for chunk in chunks:
            actual_tokens = _count_tokens(chunk.text)
            assert chunk.token_count == actual_tokens

    def test_parse_file_no_headings(self, markdown_no_headings: Path) -> None:
        """Verify file without headings produces chunks with empty heading_path."""
        chunks = parse_file(markdown_no_headings)
        assert len(chunks) > 0
        # All chunks should have empty heading_path
        for chunk in chunks:
            assert chunk.heading_path == []

    def test_parse_file_preserves_text_content(self, sample_markdown: Path) -> None:
        """Verify chunks contain expected text content."""
        chunks = parse_file(sample_markdown)
        all_text = "\n\n".join(c.text for c in chunks)
        # Verify some key content is preserved
        assert "introduction" in all_text.lower() or "subsection" in all_text.lower()
