import pytest
from pathlib import Path


@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing."""
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
    """Create a corpus of multiple markdown files for testing."""
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
    """Create an empty markdown file for edge case testing."""
    md_file = tmp_path / "empty.md"
    md_file.write_text("")
    return md_file


@pytest.fixture
def markdown_no_headings(tmp_path: Path) -> Path:
    """Create a markdown file with only paragraphs, no headings."""
    md_file = tmp_path / "no_headings.md"
    md_file.write_text(
        """Just plain text here.

No headings at all.

Just paragraphs separated by blank lines.
"""
    )
    return md_file
