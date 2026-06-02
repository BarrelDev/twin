"""Tests for the RAG pipeline end-to-end."""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from twin.llm.base import LLMProvider, LLMResponse
from twin.query.retriever import QueryResult, Retriever
from twin.rag.pipeline import RAGOutput, RAGPipeline


@pytest.fixture
def sample_chunks() -> list[QueryResult]:
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
    retriever = MagicMock(spec=Retriever)
    retriever.query.return_value = sample_chunks
    return retriever


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM provider with async complete() returning a LLMResponse."""
    llm = MagicMock(spec=LLMProvider)
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content="Decorators are functions that wrap other functions to modify their behavior."
        )
    )
    llm.extract_answer = Mock(
        return_value="Decorators are functions that wrap other functions to modify their behavior."
    )
    return llm


@pytest.fixture
def rag_pipeline(mock_retriever: MagicMock, mock_llm: MagicMock) -> RAGPipeline:
    return RAGPipeline(mock_retriever, mock_llm)


class TestRAGPipelineInitialization:

    def test_initializes_with_retriever_and_llm(
        self, mock_retriever: MagicMock, mock_llm: MagicMock
    ) -> None:
        pipeline = RAGPipeline(mock_retriever, mock_llm)
        assert pipeline._retriever is mock_retriever
        assert pipeline._llm is mock_llm


class TestRAGPipelineRetrieveContext:

    def test_retrieve_context_returns_query_results(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        result = rag_pipeline._retrieve_context("test question", k=5)
        assert isinstance(result, list)
        assert len(result) == len(sample_chunks)
        assert all(isinstance(chunk, QueryResult) for chunk in result)

    def test_retrieve_context_calls_retriever_with_correct_params(
        self, rag_pipeline: RAGPipeline
    ) -> None:
        rag_pipeline._retrieve_context("test question", k=3)
        rag_pipeline._retriever.query.assert_called_once_with("test question", k=3)

    def test_retrieve_context_uses_default_k(
        self, rag_pipeline: RAGPipeline
    ) -> None:
        rag_pipeline._retrieve_context("test question", k=5)


class TestRAGPipelineFormatContext:

    def test_format_context_returns_tuple(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        context_text, sources = rag_pipeline._format_context(sample_chunks)
        assert isinstance(context_text, str)
        assert isinstance(sources, list)

    def test_format_context_text_contains_source_markers(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        context_text, _ = rag_pipeline._format_context(sample_chunks)
        assert "[source:" in context_text
        assert "python.md" in context_text

    def test_format_context_text_contains_chunk_content(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        context_text, _ = rag_pipeline._format_context(sample_chunks)
        assert "decorators" in context_text.lower()
        assert "async" in context_text.lower()

    def test_format_context_sources_are_deduplicated(
        self, rag_pipeline: RAGPipeline
    ) -> None:
        chunks = [
            QueryResult("c1", "Text 1", "/notes/doc.md", ["Section A"], 0.9),
            QueryResult("c2", "Text 2", "/notes/doc.md", ["Section A"], 0.8),
        ]
        _, sources = rag_pipeline._format_context(chunks)
        assert len(sources) == 1


class TestRAGPipelineSynthesizeAnswer:

    @pytest.mark.anyio
    async def test_synthesize_answer_returns_string(
        self, rag_pipeline: RAGPipeline
    ) -> None:
        answer = await rag_pipeline._synthesize_answer("What are decorators?", "Context text")
        assert isinstance(answer, str)
        assert len(answer) > 0

    @pytest.mark.anyio
    async def test_synthesize_answer_calls_llm_complete(
        self, rag_pipeline: RAGPipeline
    ) -> None:
        await rag_pipeline._synthesize_answer("What are decorators?", "Context text")
        rag_pipeline._llm.complete.assert_called_once()

    @pytest.mark.anyio
    async def test_synthesize_answer_uses_system_prompt(
        self, rag_pipeline: RAGPipeline
    ) -> None:
        await rag_pipeline._synthesize_answer("What are decorators?", "Context text")
        call_args = rag_pipeline._llm.complete.call_args
        assert "system" in call_args.kwargs or len(call_args.args) >= 3

    @pytest.mark.anyio
    async def test_synthesize_answer_formats_message_correctly(
        self, rag_pipeline: RAGPipeline
    ) -> None:
        context = "Important context text"
        question = "What is this?"
        await rag_pipeline._synthesize_answer(question, context)
        call_args = rag_pipeline._llm.complete.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs.get("messages")
        assert messages is not None
        message_content = messages[0]["content"]
        assert context in message_content
        assert question in message_content


class TestRAGPipelineQuery:

    @pytest.mark.anyio
    async def test_query_returns_rag_output(self, rag_pipeline: RAGPipeline) -> None:
        result = await rag_pipeline.query("What are decorators?")
        assert isinstance(result, RAGOutput)

    @pytest.mark.anyio
    async def test_query_includes_answer(self, rag_pipeline: RAGPipeline) -> None:
        result = await rag_pipeline.query("What are decorators?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    @pytest.mark.anyio
    async def test_query_includes_sources(self, rag_pipeline: RAGPipeline) -> None:
        result = await rag_pipeline.query("What are decorators?")
        assert isinstance(result.sources, list)
        assert len(result.sources) > 0
        for source in result.sources:
            assert "path" in source
            assert "heading_path" in source

    @pytest.mark.anyio
    async def test_query_includes_context_chunks(self, rag_pipeline: RAGPipeline) -> None:
        result = await rag_pipeline.query("What are decorators?")
        assert isinstance(result.context_chunks, list)
        assert len(result.context_chunks) > 0
        assert all(isinstance(c, QueryResult) for c in result.context_chunks)

    @pytest.mark.anyio
    async def test_query_respects_k_parameter(self, rag_pipeline: RAGPipeline) -> None:
        await rag_pipeline.query("What are decorators?", k=3)
        rag_pipeline._retriever.query.assert_called_with("What are decorators?", k=3)

    @pytest.mark.anyio
    async def test_query_with_default_k_parameter(self, rag_pipeline: RAGPipeline) -> None:
        await rag_pipeline.query("What are decorators?")
        rag_pipeline._retriever.query.assert_called_with("What are decorators?", k=5)


class TestRAGPipelineIntegration:

    @pytest.mark.anyio
    async def test_end_to_end_full_pipeline(self, rag_pipeline: RAGPipeline) -> None:
        result = await rag_pipeline.query("What are decorators?")
        assert isinstance(result, RAGOutput)
        assert result.answer
        assert result.sources
        assert result.context_chunks

    @pytest.mark.anyio
    async def test_sources_match_retrieved_chunks(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[QueryResult]
    ) -> None:
        result = await rag_pipeline.query("What are decorators?")
        chunk_filenames = {c.source_path.split("/")[-1] for c in result.context_chunks}
        for source in result.sources:
            assert source["path"] in chunk_filenames

    @pytest.mark.anyio
    async def test_pipeline_with_empty_retrieval(self, rag_pipeline: RAGPipeline) -> None:
        rag_pipeline._retriever.query.return_value = []
        result = await rag_pipeline.query("What are decorators?")
        assert isinstance(result, RAGOutput)
        assert result.context_chunks == []
        assert result.sources == []

    @pytest.mark.anyio
    async def test_pipeline_orchestration_order(self, rag_pipeline: RAGPipeline) -> None:
        """Pipeline calls retrieve → format → synthesize in that order."""
        call_order = []
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

        # _synthesize_answer is async; the lambda returns its coroutine.
        rag_pipeline._synthesize_answer = lambda q, c: (
            call_order.append("synthesize"),
            original_synthesize(q, c),
        )[1]

        await rag_pipeline.query("Test question")
        assert call_order == ["retrieve", "format", "synthesize"]
