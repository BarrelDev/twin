import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from twin_core import (
    Chunk,
    chunk_text as _rust_chunk_text,
    count_tokens as _rust_count_tokens,
    parse_markdown_structure as _rust_parse_markdown_structure,
    split_section_into_chunks as _rust_split_section_into_chunks,
)

MAX_TOKENS = 512
OVERLAP = 64

def _count_tokens(text: str) -> int:
    """Count approximate tokens in text.

    Args:
        text: Text to count tokens for.

    Returns:
        Approximate token count.
    """
    return _rust_count_tokens(text)

def _extract_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown content.

    Looks for YAML block between --- delimiters at the start of the file.

    Args:
        content: Raw markdown content.

    Returns:
        Tuple of (frontmatter_dict, remaining_content).
    """
    if not content.startswith("---"):
        return {}, content

    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not match:
        return {}, content

    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, content

    body = content[match.end() :]
    return frontmatter, body

def _parse_markdown_structure(content: str) -> list[tuple[list[str], str]]:
    """Parse markdown content into heading-based sections.

    Args:
        content: Markdown content to parse.

    Returns:
        List of (heading_path, content) tuples.
    """
    return _rust_parse_markdown_structure(content)

def _split_section_into_chunks(
    text: str, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP
) -> list[str]:
    """Split text into chunks respecting token budget and overlap.

    Args:
        text: Text to chunk.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Tokens to overlap between chunks.

    Returns:
        List of text chunks.
    """
    return _rust_split_section_into_chunks(text, max_tokens, overlap_tokens)

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
    if not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    if not content.strip():
        return []

    # Extract frontmatter
    frontmatter, body = _extract_frontmatter(content)

    # Use doc_id from frontmatter or derive from filename
    doc_id = str(frontmatter.get("id", path.stem))

    return _rust_chunk_text(body, doc_id, str(path), MAX_TOKENS, OVERLAP)
