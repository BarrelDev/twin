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
    pass


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
    pass


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
    pass
