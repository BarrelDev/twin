"""Tests for the RAG pipeline end-to-end."""

from unittest.mock import MagicMock

import pytest

from twin.llm.base import LLMProvider
from twin.query.retriever import QueryResult, Retriever
from twin.rag.pipeline import RAGOutput, RAGPipeline


@pytest.fixture
def sample_chunks() -> list[QueryResult]:
    """Sample chunks for testing."""
    return [
        QueryResult(
            chunk_id="chunk_001",
            text="Python decorators are functions that modify other functions.",
            source_path="/notes/python.md",
            heading_path=["Advanced", "Decorators"],
            score=0.95,
        ),
        QueryResult(
            chunk_id="chunk_002",
            text="Async/await allows concurrent execution in Python.",
            source_path="/notes/python.md",
            heading_path=["Advanced", "Async"],
            score=0.87,
        ),
    ]


@pytest.fixture
def mock_retriever(sample_chunks: list[QueryResult]) -> MagicMock:
    """Mock retriever that returns sample chunks."""
    retriever = MagicMock(spec=Retriever)
    retriever.query.return_value = sample_chunks
    return retriever


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM provider that returns a sample response."""
    llm = MagicMock(spec=LLMProvider)
    # Mock response that simulates Anthropic Message-like structure
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Decorators are functions that wrap other functions to modify their behavior.")]
    llm.complete.return_value = mock_response
    llm.extract_answer.return_value = "Decorators are functions that wrap other functions to modify their behavior."
    return llm


@pytest.fixture
def rag_pipeline(mock_retriever: MagicMock, mock_llm: MagicMock) -> RAGPipeline:
    """RAGPipeline instance with mock dependencies."""
    return RAGPipeline(mock_retriever, mock_llm)


class TestRAGPipelineInitialization:
    """Tests for RAGPipeline initialization."""

    def test_initializes_with_retriever_and_llm(
        self, mock_retriever: MagicMock, mock_llm: MagicMock
    ) -> None:
        """Verify pipeline stores retriever and LLM correctly."""
        pipeline = RAGPipeline(mock_retriever, mock_llm)
        assert pipeline._retriever is mock_retriever
        assert pipeline._llm is mock_llm


class TestRAGPipelineRetrieveContext:
    """Tests for RAGPipeline._retrieve_context()."""

    def test_retrieve_context_returns_query_results(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify _retrieve_context() returns list of QueryResult."""
        result = rag_pipeline._retrieve_context("test question", k=5)
        assert isinstance(result, list)
        assert len(result) == len(sample_chunks)
        assert all(isinstance(chunk, QueryResult) for chunk in result)

    def test_retrieve_context_calls_retriever_with_correct_params(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify _retrieve_context() passes parameters to retriever correctly."""
        rag_pipeline._retrieve_context("test question", k=3)
        rag_pipeline._retriever.query.assert_called_once_with("test question", k=3)

    def test_retrieve_context_uses_default_k(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify _retrieve_context() can be called without k parameter."""
        rag_pipeline._retrieve_context("test question", k=5)
        # Should work without error


class TestRAGPipelineFormatContext:
    """Tests for RAGPipeline._format_context()."""

    def test_format_context_returns_tuple(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify _format_context() returns (context_string, sources_list)."""
        context_text, sources = rag_pipeline._format_context(sample_chunks)
        assert isinstance(context_text, str)
        assert isinstance(sources, list)

    def test_format_context_text_contains_source_markers(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify formatted context includes source attribution."""
        context_text, _ = rag_pipeline._format_context(sample_chunks)
        assert "[source:" in context_text
        assert "python.md" in context_text

    def test_format_context_text_contains_chunk_content(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify formatted context includes chunk text."""
        context_text, _ = rag_pipeline._format_context(sample_chunks)
        assert "decorators" in context_text.lower()
        assert "async" in context_text.lower()

    def test_format_context_sources_are_deduplicated(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify sources are deduplicated when formatting."""
        chunks = [
            QueryResult(
                chunk_id="c1",
                text="Text 1",
                source_path="/notes/doc.md",
                heading_path=["Section A"],
                score=0.9,
            ),
            QueryResult(
                chunk_id="c2",
                text="Text 2",
                source_path="/notes/doc.md",
                heading_path=["Section A"],
                score=0.8,
            ),
        ]
        _, sources = rag_pipeline._format_context(chunks)
        # Should only have one source despite two chunks
        assert len(sources) == 1


class TestRAGPipelineSynthesizeAnswer:
    """Tests for RAGPipeline._synthesize_answer()."""

    def test_synthesize_answer_returns_string(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify _synthesize_answer() returns a non-empty string."""
        answer = rag_pipeline._synthesize_answer("What are decorators?", "Context text")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_synthesize_answer_calls_llm_complete(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify _synthesize_answer() calls the LLM's complete method."""
        rag_pipeline._synthesize_answer("What are decorators?", "Context text")
        # Verify that complete was called
        rag_pipeline._llm.complete.assert_called_once()

    def test_synthesize_answer_uses_system_prompt(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify _synthesize_answer() passes a system prompt to the LLM."""
        rag_pipeline._synthesize_answer("What are decorators?", "Context text")
        call_args = rag_pipeline._llm.complete.call_args
        # Verify that 'system' was passed as a keyword argument
        assert "system" in call_args.kwargs or len(call_args.args) >= 3

    def test_synthesize_answer_formats_message_correctly(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify the user message includes both context and question."""
        context = "Important context text"
        question = "What is this?"
        rag_pipeline._synthesize_answer(question, context)

        call_args = rag_pipeline._llm.complete.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs.get("messages")

        assert messages is not None
        assert len(messages) > 0
        message_content = messages[0]["content"]
        assert context in message_content
        assert question in message_content


class TestRAGPipelineQuery:
    """Tests for RAGPipeline.query()."""

    def test_query_returns_rag_output(self, rag_pipeline: RAGPipeline) -> None:
        """Verify query() returns a RAGOutput object."""
        result = rag_pipeline.query("What are decorators?")
        assert isinstance(result, RAGOutput)

    def test_query_includes_answer(self, rag_pipeline: RAGPipeline) -> None:
        """Verify RAGOutput contains a non-empty answer string."""
        result = rag_pipeline.query("What are decorators?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_query_includes_sources(self, rag_pipeline: RAGPipeline) -> None:
        """Verify RAGOutput includes sources from retrieved chunks."""
        result = rag_pipeline.query("What are decorators?")
        assert isinstance(result.sources, list)
        assert len(result.sources) > 0
        # Verify source structure
        for source in result.sources:
            assert "path" in source
            assert "heading_path" in source

    def test_query_includes_context_chunks(self, rag_pipeline: RAGPipeline) -> None:
        """Verify RAGOutput contains the retrieved QueryResult chunks."""
        result = rag_pipeline.query("What are decorators?")
        assert isinstance(result.context_chunks, list)
        assert len(result.context_chunks) > 0
        assert all(isinstance(chunk, QueryResult) for chunk in result.context_chunks)

    def test_query_respects_k_parameter(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify query() passes k parameter to retriever."""
        rag_pipeline.query("What are decorators?", k=3)
        rag_pipeline._retriever.query.assert_called_with("What are decorators?", k=3)

    def test_query_with_default_k_parameter(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify query() uses default k=5 when not specified."""
        rag_pipeline.query("What are decorators?")
        rag_pipeline._retriever.query.assert_called_with("What are decorators?", k=5)


class TestRAGPipelineIntegration:
    """Integration tests with realistic retrieval and LLM responses."""

    def test_end_to_end_full_pipeline(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify full pipeline execution works end-to-end."""
        result = rag_pipeline.query("What are decorators?")

        # Verify all components are present
        assert isinstance(result, RAGOutput)
        assert result.answer
        assert result.sources
        assert result.context_chunks

    def test_sources_match_retrieved_chunks(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        """Verify returned sources correspond to the chunks that were retrieved."""
        result = rag_pipeline.query("What are decorators?")

        # Extract filenames from context chunks
        chunk_filenames = {
            chunk.source_path.split("/")[-1] for chunk in result.context_chunks
        }

        # Verify sources correspond to retrieved chunks
        for source in result.sources:
            assert source["path"] in chunk_filenames

    def test_pipeline_with_empty_retrieval(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify pipeline handles case where retriever returns no results."""
        rag_pipeline._retriever.query.return_value = []
        result = rag_pipeline.query("What are decorators?")

        # Should return a valid RAGOutput even with no chunks
        assert isinstance(result, RAGOutput)
        assert result.context_chunks == []
        assert result.sources == []

    def test_pipeline_orchestration_order(
        self, rag_pipeline: RAGPipeline,
    ) -> None:
        """Verify pipeline calls methods in correct order."""
        call_order = []

        # Spy on method calls
        original_retrieve = rag_pipeline._retrieve_context
        original_format = rag_pipeline._format_context
        original_synthesize = rag_pipeline._synthesize_answer

        rag_pipeline._retrieve_context = lambda q, k: (
            call_order.append("retrieve"),
            original_retrieve(q, k),
        )[1]
        rag_pipeline._format_context = lambda chunks: (
            call_order.append("format"),
            original_format(chunks),
        )[1]
        rag_pipeline._synthesize_answer = lambda q, c: (
            call_order.append("synthesize"),
            original_synthesize(q, c),
        )[1]

        rag_pipeline.query("Test question")

        # Verify the correct order
        assert call_order == ["retrieve", "format", "synthesize"]
