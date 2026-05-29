"""RAG (Retrieval-Augmented Generation) pipeline orchestration."""

from dataclasses import dataclass

from twin.llm.base import LLMProvider
from twin.query.retriever import QueryResult, Retriever


@dataclass
class RAGOutput:
    """Output from the RAG pipeline."""

    answer: str
    """The synthesized answer from the LLM."""

    sources: list[dict]
    """
    Deduplicated list of sources consulted.
    Each item is a dict with 'path' (filename) and 'heading_path' (breadcrumb).
    """

    context_chunks: list[QueryResult]
    """The retrieved chunks that were used as context."""


class RAGPipeline:
    """
    Orchestrates the RAG pipeline: retrieve → format → generate → return.

    The pipeline takes a user query, retrieves relevant chunks from the knowledge
    base, formats them as context, and synthesizes an answer using an LLM.
    """

    def __init__(self, retriever: Retriever, llm: LLMProvider) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            retriever: Retriever instance for knowledge base search.
            llm: LLMProvider instance for generating answers.
        """
        pass

    def query(self, question: str, k: int = 5) -> RAGOutput:
        """
        Execute the RAG pipeline end-to-end.

        Steps:
        1. Retrieve top-k chunks relevant to the question.
        2. Format chunks as context with source attribution.
        3. Call LLM with question and context.
        4. Extract sources from retrieved chunks.
        5. Return synthesized answer with sources.

        Args:
            question: The user's question or query string.
            k: Number of chunks to retrieve (default: 5).

        Returns:
            RAGOutput with answer, sources, and context_chunks.
        """
        pass

    def _retrieve_context(self, question: str, k: int) -> list[QueryResult]:
        """
        Retrieve relevant chunks from the knowledge base.

        Args:
            question: The user's question string.
            k: Number of chunks to retrieve.

        Returns:
            List of QueryResult ordered by relevance.
        """
        pass

    def _format_context(self, chunks: list[QueryResult]) -> tuple[str, list[dict]]:
        """
        Format chunks as context and extract sources.

        Args:
            chunks: List of retrieved QueryResult.

        Returns:
            Tuple of (formatted_context_string, sources_list).
        """
        pass

    def _synthesize_answer(
        self, question: str, context: str
    ) -> str:
        """
        Call the LLM to synthesize an answer from context.

        Args:
            question: The original user question.
            context: Formatted context with attributed chunks.

        Returns:
            The LLM's synthesized answer.
        """
        pass
