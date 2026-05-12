from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """A semantic chunk extracted from a document."""

    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    heading_path: list[str]
    chunk_index: int
    token_count: int


def parse_file(path: Path) -> list[Chunk]:
    """
    Parse a markdown file into semantic chunks.

    Splits on markdown headings (primary) and paragraph breaks (secondary).
    Chunks overlap by SECONDBRAIN_OVERLAP tokens to preserve context.

    Args:
        path: Path to markdown file.

    Returns:
        List of Chunk objects extracted from the file.
    """
    return []
