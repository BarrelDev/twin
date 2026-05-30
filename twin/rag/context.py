"""Context formatting for RAG pipeline with source attribution."""

from dataclasses import dataclass
from pathlib import Path

from twin.query.retriever import QueryResult


@dataclass
class FormattedContext:
    """Formatted context for LLM consumption."""

    text: str
    """The formatted context string ready for the LLM."""

    sources: list[dict]
    """Deduplicated list of sources. Each item has 'path' and 'heading_path' keys."""


def format_chunks_as_context(chunks: list[QueryResult]) -> str:
    """
    Format retrieved chunks as a context block with source attribution.

    Each chunk is prefixed with its source (filename + heading path) so the LLM
    knows where information came from. Chunks are separated by blank lines.

    Args:
        chunks: List of QueryResult from the retriever, ordered by relevance.

    Returns:
        Formatted context string ready to pass to the LLM.
    """
    formatted_chunks = []
    for c in chunks:
        filename = Path(c.source_path).name
        breadcrumb = " > ".join([filename] + c.heading_path)
        source_line = f"[source: {breadcrumb}]\n"
        formatted_chunks.append(source_line + c.text)

    return "\n\n".join(formatted_chunks)


def extract_sources(chunks: list[QueryResult]) -> list[dict]:
    """
    Extract and deduplicate sources from a list of chunks.

    Returns a deduplicated list of sources, preserving order of first appearance.
    Each source dict contains 'path' (just the filename) and 'heading_path' (the
    full breadcrumb).

    Args:
        chunks: List of QueryResult from the retriever.

    Returns:
        List of dicts with 'path' and 'heading_path' keys, deduplicated.
    """
    seen = set()
    sources = []
    for c in chunks:
        path = Path(c.source_path).name
        heading = tuple(c.heading_path)
        key = (path, heading)
        if key not in seen:
            seen.add(key)
            sources.append({"path": path, "heading_path": c.heading_path})
    
    return sources


def prepare_rag_context(chunks: list[QueryResult]) -> FormattedContext:
    """
    Prepare context for RAG by formatting chunks and extracting sources.

    Orchestrates format_chunks_as_context() and extract_sources() into a single
    FormattedContext object.

    Args:
        chunks: List of QueryResult from the retriever.

    Returns:
        FormattedContext with 'text' and 'sources' fields.
    """
    text = format_chunks_as_context(chunks)
    sources = extract_sources(chunks)
    return FormattedContext(text=text, sources=sources)
