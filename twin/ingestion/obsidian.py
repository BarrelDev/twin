import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from twin.config import AppConfig

try:
    from twin_core import (
        count_tokens as _rust_count_tokens,
        parse_markdown_structure as _rust_parse_markdown_structure,
        split_section_into_chunks as _rust_split,
    )
    _USE_RUST = True
except ImportError:
    _USE_RUST = False

# Captures the note name from [[Note Name]] and [[Note Name|Alias]]
WIKILINK_RE = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
# Replaces [[Note|Alias]] → Alias, [[Note]] → Note (for plain-text conversion)
WIKILINK_REPLACE_RE = re.compile(r'\[\[(?:[^\]|]+\|)?([^\]]+)\]\]')
# Matches embed directives ![[image.png]]
EMBED_RE = re.compile(r'!\[\[[^\]]+\]\]')
# Matches Obsidian tags including nested (#parent/child)
TAG_RE = re.compile(r'#([\w/]+)')


@dataclass
class _Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    heading_path: list[str]
    chunk_index: int
    token_count: int
    link_targets: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def _count_tokens(text: str) -> int:
    if _USE_RUST:
        return _rust_count_tokens(text)
    return len(text.split())


def _extract_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from Markdown content.

    Args:
        content: Raw Markdown content.

    Returns:
        Tuple of (frontmatter_dict, remaining_body).
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
    return frontmatter, content[match.end():]


def parse_obsidian_file(
    path: Path, config: AppConfig
) -> tuple[list[_Chunk], dict[str, Any]]:
    """
    Parse an Obsidian Markdown file into chunks with metadata.

    Extracts wikilinks and tags from the full document, strips embeds
    and converts wikilinks to plain text before chunking so that embedded
    images and link syntax are not included in the embedded representation.

    Args:
        path: Path to the Obsidian .md file.
        config: AppConfig with chunk_tokens and overlap_tokens settings.

    Returns:
        Tuple of (chunks, obsidian_metadata) where obsidian_metadata contains:
          - link_targets: list[str] — wikilink note names (not aliases), deduplicated
          - tags: list[str] — Obsidian tags including nested (#parent/child), deduplicated
          - frontmatter: dict — all YAML frontmatter fields

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Obsidian file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Extract wikilink note names from raw text (before any cleaning), deduplicated
    link_targets = list(dict.fromkeys(WIKILINK_RE.findall(text)))

    # Parse frontmatter before extracting tags so YAML values don't pollute tag list
    frontmatter, body = _extract_frontmatter(text)

    # Extract tags from body only
    tags = list(dict.fromkeys(TAG_RE.findall(body)))

    # Strip embed directives before wikilink replacement
    clean_body = EMBED_RE.sub("", body)
    # Convert wikilinks to plain text: [[Note|Alias]] → Alias, [[Note]] → Note
    clean_body = WIKILINK_REPLACE_RE.sub(r"\1", clean_body)

    doc_id = str(frontmatter.get("id", path.stem))
    chunks = _chunk_markdown(clean_body, doc_id, path, config, link_targets, tags)

    obsidian_meta: dict[str, Any] = {
        "link_targets": link_targets,
        "tags": tags,
        "frontmatter": frontmatter,
    }
    return chunks, obsidian_meta


def _chunk_markdown(
    text: str,
    doc_id: str,
    path: Path,
    config: AppConfig,
    link_targets: list[str],
    tags: list[str],
) -> list[_Chunk]:
    """
    Split cleaned Markdown text into chunks using heading-aware splitting.

    Each chunk carries the document-level link_targets and tags so that
    retrieval results expose Obsidian graph context.

    Args:
        text: Cleaned Markdown body with frontmatter, embeds, and wikilinks removed.
        doc_id: Document identifier used for chunk IDs.
        path: Source file path for attribution.
        config: AppConfig for chunk size settings.
        link_targets: Document-level wikilink note names to attach to every chunk.
        tags: Document-level Obsidian tags to attach to every chunk.

    Returns:
        List of _Chunk objects, empty if text has no content.
    """
    if not text.strip():
        return []

    if _USE_RUST:
        sections: list[tuple[list[str], str]] = _rust_parse_markdown_structure(text)
    else:
        sections = [([path.stem], text)]

    chunks: list[_Chunk] = []
    chunk_index = 0

    for heading_path, section_content in sections:
        if _USE_RUST:
            section_texts = _rust_split(section_content, config.chunk_tokens, config.overlap_tokens)
        else:
            section_texts = _simple_split(section_content, config.chunk_tokens)

        for chunk_text in section_texts:
            if not chunk_text.strip():
                continue
            chunks.append(
                _Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index}",
                    doc_id=doc_id,
                    text=chunk_text,
                    source_path=str(path),
                    heading_path=heading_path,
                    chunk_index=chunk_index,
                    token_count=_count_tokens(chunk_text),
                    link_targets=link_targets,
                    tags=tags,
                )
            )
            chunk_index += 1

    return chunks


def _simple_split(text: str, max_tokens: int) -> list[str]:
    """Fallback paragraph-based splitter used when twin_core is unavailable.

    Args:
        text: Text to split.
        max_tokens: Approximate maximum tokens per chunk (word count used as proxy).

    Returns:
        List of text chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(para.split())
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks
