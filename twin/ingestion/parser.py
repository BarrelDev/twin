import re
from dataclasses import dataclass
from pathlib import Path

import yaml

MAX_TOKENS = 512
OVERLAP = 64


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


def _count_tokens(text: str) -> int:
    """Estimate token count as word count (rough approximation)."""
    return len(text.split())


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
    """Extract heading hierarchy and associated content sections.

    Splits content by markdown headings (primary boundary) and extracts
    the text under each heading as a section.

    Args:
        content: Markdown body content (without frontmatter).

    Returns:
        List of (heading_path, section_text) tuples.
    """
    if not content.strip():
        return []

    sections: list[tuple[list[str], str]] = []
    current_path: list[str] = []
    current_text: list[str] = []

    # Split by heading lines (# pattern at line start)
    lines = content.split("\n")
    heading_pattern = re.compile(r"^(#+)\s+(.+)$")

    for line in lines:
        match = heading_pattern.match(line)
        if match:
            # Save previous section if it has content
            text = "\n".join(current_text).strip()
            if text:
                sections.append((current_path.copy(), text))
            current_text = []

            # Update heading path based on level
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            current_path = current_path[: level - 1]
            current_path.append(heading_text)
        else:
            current_text.append(line)

    # Save final section
    text = "\n".join(current_text).strip()
    if text:
        sections.append((current_path, text))

    return sections if sections else [([], content.strip())]


def _split_section_into_chunks(
    text: str, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP
) -> list[str]:
    """Split a section into chunks respecting token budget.

    Primary split on paragraphs (blank lines). If a single paragraph
    exceeds the budget, splits on sentences.

    Args:
        text: Section text to chunk.
        max_tokens: Maximum tokens per chunk (includes overlap).
        overlap_tokens: Tokens to overlap between chunks.

    Returns:
        List of chunk texts with overlap applied.
    """
    if not text.strip():
        return []

    # Reduce effective budget to account for overlap
    effective_budget = max_tokens - overlap_tokens if overlap_tokens > 0 else max_tokens

    # Split by paragraphs (blank lines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # If paragraph alone exceeds budget, split by sentences
        if para_tokens > effective_budget:
            # Save current chunk if non-empty
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_tokens = _count_tokens(sent)
                if sent_tokens > effective_budget:
                    # Very long sentence; include as-is (will exceed budget)
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                    chunks.append(sent)
                    current_tokens = 0
                elif current_tokens + sent_tokens > effective_budget:
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    current_chunk = [sent]
                    current_tokens = sent_tokens
                else:
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
        else:
            # Paragraph fits; check if adding it exceeds budget
            if current_tokens + para_tokens > effective_budget:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

    # Save final chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Apply overlap if multiple chunks exist
    if len(chunks) <= 1 or overlap_tokens == 0:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        curr_chunk = chunks[i]

        # Extract last ~overlap_tokens from previous chunk
        prev_sentences = re.split(r"(?<=[.!?])\s+", prev_chunk)
        overlap_text: list[str] = []
        overlap_count = 0

        for sent in reversed(prev_sentences):
            sent_tokens = _count_tokens(sent)
            if overlap_count + sent_tokens <= overlap_tokens:
                overlap_text.insert(0, sent)
                overlap_count += sent_tokens
            else:
                break

        if overlap_text:
            combined = " ".join(overlap_text) + "\n\n" + curr_chunk
            overlapped.append(combined)
        else:
            overlapped.append(curr_chunk)

    return overlapped


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

    # Parse markdown structure
    sections = _parse_markdown_structure(body)

    chunks: list[Chunk] = []
    chunk_index = 0

    for heading_path, section_text in sections:
        # Split section into chunks
        text_chunks = _split_section_into_chunks(section_text)

        for chunk_text in text_chunks:
            if not chunk_text.strip():
                continue

            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            token_count = _count_tokens(chunk_text)

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                source_path=str(path),
                heading_path=heading_path,
                chunk_index=chunk_index,
                token_count=token_count,
            )
            chunks.append(chunk)
            chunk_index += 1

    return chunks
