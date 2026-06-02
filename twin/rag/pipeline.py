"""RAG (Retrieval-Augmented Generation) pipeline orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator
from datetime import datetime, timezone

from twin.llm.base import LLMProvider
from twin.query.retriever import QueryResult, Retriever
from twin.rag.prompts import SystemPrompts
from twin.rag.context import prepare_rag_context
from twin.usage import UsageLogger, UsageRecord

_file_logger = UsageLogger(Path("~/.twin").expanduser())


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
    """The retrieved chunks used as context."""


class RAGPipeline:
    """
    Orchestrates the RAG pipeline: retrieve → format → generate → return.

    The pipeline takes a user query, retrieves relevant chunks from the
    knowledge base, formats them as context, and synthesizes an answer
    using an LLM.
    """

    def __init__(self, retriever: Retriever, llm: LLMProvider) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            retriever: Retriever instance for knowledge base search.
            llm: LLMProvider instance for generating answers.
        """
        self._retriever = retriever
        self._llm = llm
        self._session_records: list[UsageRecord] = []

    @property
    def session_records(self) -> list[UsageRecord]:
        """Usage records accumulated during this pipeline session."""
        return self._session_records

    async def query(self, question: str, k: int = 5) -> RAGOutput:
        """
        Execute the RAG pipeline end-to-end.

        Steps:
        1. Retrieve top-k chunks relevant to the question.
        2. Format chunks as context with source attribution.
        3. Call LLM to synthesize an answer.
        4. Return the answer with sources and context chunks.

        Args:
            question: The user's question or query string.
            k: Number of chunks to retrieve (default: 5).

        Returns:
            RAGOutput with answer, sources, and context_chunks.
        """
        chunks = self._retrieve_context(question, k)
        context_text, sources = self._format_context(chunks)
        answer = await self._synthesize_answer(question, context_text)
        return RAGOutput(answer=answer, sources=sources, context_chunks=chunks)

    def _retrieve_context(self, question: str, k: int) -> list[QueryResult]:
        """
        Retrieve relevant chunks from the knowledge base.

        Args:
            question: The user's question string.
            k: Number of chunks to retrieve.

        Returns:
            List of QueryResult ordered by relevance.
        """
        return self._retriever.query(question, k=k)

    def _format_context(self, chunks: list[QueryResult]) -> tuple[str, list[dict]]:
        """
        Format chunks as context and extract deduplicated sources.

        Args:
            chunks: List of retrieved QueryResult.

        Returns:
            Tuple of (formatted_context_string, sources_list).
        """
        formatted = prepare_rag_context(chunks)
        return formatted.text, formatted.sources

    async def query_stream(
        self, question: str, k: int = 5
    ) -> tuple[AsyncGenerator[str, None], list[dict]]:
        """
        Execute the RAG pipeline with token-by-token streaming.

        Sources are determined synchronously during retrieval (before the LLM
        call) so the caller has them immediately. The returned async generator
        yields text tokens as they arrive from the provider.

        A streaming UsageRecord (with 0 tokens) is appended to session_records
        when the generator is fully consumed, since token counts are unavailable
        from the streaming API.

        Args:
            question: The user's question or query string.
            k: Number of chunks to retrieve (default: 5).

        Returns:
            Tuple of (token_stream, sources) where token_stream is an async
            generator yielding text chunks and sources is the deduplicated
            list of source attribution dicts.
        """
        chunks = self._retrieve_context(question, k)
        context_text, sources = self._format_context(chunks)
        usr_message = f"Context:\n{context_text}\n\nQuestion: {question}"
        messages = [{"role": "user", "content": usr_message}]
        llm = self._llm
        session_records = self._session_records
        provider = getattr(self._llm, "provider_name", "unknown")
        model_name = getattr(self._llm, "model", "unknown")

        async def _token_stream() -> AsyncGenerator[str, None]:
            async for token in llm.stream(messages, system=SystemPrompts.RAG_SYSTEM):
                yield token
            # Stream complete — log the call (token counts unavailable for streaming)
            session_records.append(UsageRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                command="rag",
                provider=provider,
                model=model_name,
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost_usd=None,
            ))

        return _token_stream(), sources

    async def _synthesize_answer(self, question: str, context: str) -> str:
        """
        Call the LLM to synthesize an answer from context.

        Args:
            question: The original user question.
            context: Formatted context with attributed chunks.

        Returns:
            The LLM's synthesized answer as a string.
        """
        usr_message = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [{"role": "user", "content": usr_message}]
        response = await self._llm.complete(messages, tools=None, system=SystemPrompts.RAG_SYSTEM)

        # Log usage — best-effort; skip if token counts are not available (e.g. in tests)
        try:
            record = UsageRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                command="rag",
                provider=getattr(self._llm, "provider_name", "unknown"),
                model=getattr(self._llm, "model", "unknown"),
                prompt_tokens=int(response.prompt_tokens or 0),
                completion_tokens=int(response.completion_tokens or 0),
                estimated_cost_usd=self._llm.estimate_cost(
                    response.prompt_tokens or 0, response.completion_tokens or 0
                ),
            )
            self._session_records.append(record)
            _file_logger.log(record)
        except (TypeError, ValueError, OSError):
            pass

        return self._llm.extract_answer(response)
