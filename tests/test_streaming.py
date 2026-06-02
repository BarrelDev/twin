"""Tests for streaming output in the RAG pipeline and agent runtime."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from twin.llm.base import LLMResponse, ToolCall


# ── RAG pipeline streaming ───────────────────────────────────────────────────


@pytest.mark.anyio
async def test_rag_query_stream_yields_tokens(
    mock_llm_provider, tmp_lance_db, mock_embedder
) -> None:
    """query_stream() yields tokens from the provider's stream() method."""
    from twin.query.retriever import Retriever
    from twin.rag.pipeline import RAGPipeline

    mock_llm_provider.stream_chunks = ["Hello", " ", "world"]
    retriever = Retriever(tmp_lance_db, mock_embedder)
    pipeline = RAGPipeline(retriever, mock_llm_provider)

    stream_gen, _ = await pipeline.query_stream("test question")
    tokens = [t async for t in stream_gen]

    assert "".join(tokens) == "Hello world"


@pytest.mark.anyio
async def test_rag_query_stream_sources_returned_before_streaming(
    mock_llm_provider, tmp_lance_db, mock_embedder
) -> None:
    """Sources list is available immediately when query_stream() returns."""
    from twin.query.retriever import Retriever
    from twin.rag.pipeline import RAGPipeline

    retriever = Retriever(tmp_lance_db, mock_embedder)
    pipeline = RAGPipeline(retriever, mock_llm_provider)

    _, sources = await pipeline.query_stream("test question")

    assert isinstance(sources, list)


@pytest.mark.anyio
async def test_rag_query_stream_uses_stream_not_complete(
    mock_llm_provider, tmp_lance_db, mock_embedder
) -> None:
    """query_stream() calls provider.stream(), not provider.complete()."""
    from twin.query.retriever import Retriever
    from twin.rag.pipeline import RAGPipeline

    retriever = Retriever(tmp_lance_db, mock_embedder)
    pipeline = RAGPipeline(retriever, mock_llm_provider)

    stream_gen, _ = await pipeline.query_stream("some question", k=3)
    _ = [t async for t in stream_gen]  # consume the stream

    methods_called = [c["method"] for c in mock_llm_provider.calls]
    assert "stream" in methods_called
    assert "complete" not in methods_called


@pytest.mark.anyio
async def test_rag_query_stream_empty_kb(
    mock_llm_provider, tmp_lance_db, mock_embedder
) -> None:
    """query_stream() works when the knowledge base has no chunks."""
    from twin.query.retriever import Retriever
    from twin.rag.pipeline import RAGPipeline

    mock_llm_provider.stream_chunks = ["No context available."]
    retriever = Retriever(tmp_lance_db, mock_embedder)
    pipeline = RAGPipeline(retriever, mock_llm_provider)

    stream_gen, sources = await pipeline.query_stream("anything")
    tokens = [t async for t in stream_gen]

    assert "".join(tokens) == "No context available."
    assert sources == []


# ── Agent runtime streaming ──────────────────────────────────────────────────


class _NoOpDispatcher:
    """Minimal dispatcher that returns empty tool definitions and a canned result."""

    def get_tool_definitions(self) -> list:
        return []

    def dispatch(self, name: str, tool_input: dict) -> str:
        return "tool result"


@pytest.mark.anyio
async def test_agent_execute_stream_yields_token_events(
    mock_llm_provider,
) -> None:
    """execute_stream() yields token events for the streamed final answer."""
    from twin.agent.runtime import AgentRuntime

    mock_llm_provider.stream_chunks = ["Hello", " ", "world"]
    runtime = AgentRuntime(mock_llm_provider, _NoOpDispatcher())

    events = [ev async for ev in runtime.execute_stream("test task")]

    token_events = [ev for ev in events if ev["type"] == "token"]
    assert "".join(ev["text"] for ev in token_events) == "Hello world"


@pytest.mark.anyio
async def test_agent_execute_stream_done_event(
    mock_llm_provider,
) -> None:
    """execute_stream() always yields a single 'done' event as the last event."""
    from twin.agent.runtime import AgentRuntime

    runtime = AgentRuntime(mock_llm_provider, _NoOpDispatcher())

    events = [ev async for ev in runtime.execute_stream("test task")]

    done_events = [ev for ev in events if ev["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[-1] is events[-1], "done must be the final event"


@pytest.mark.anyio
async def test_agent_execute_stream_done_event_has_tool_count(
    mock_llm_provider,
) -> None:
    """The 'done' event carries the tool_calls count and activity_log."""
    from twin.agent.runtime import AgentRuntime

    runtime = AgentRuntime(mock_llm_provider, _NoOpDispatcher())

    events = [ev async for ev in runtime.execute_stream("test task")]
    done = next(ev for ev in events if ev["type"] == "done")

    assert "tool_calls" in done
    assert "activity_log" in done
    assert isinstance(done["activity_log"], list)


@pytest.mark.anyio
async def test_agent_execute_stream_tool_call_events(
    mock_llm_provider,
) -> None:
    """execute_stream() emits tool_call events before the final token stream."""
    from twin.agent.runtime import AgentRuntime

    call_count = 0
    tool_response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc_1", name="search_knowledge_base", input={"query": "test"})],
        stop_reason="tool_use",
    )
    final_response = LLMResponse(content="final answer")

    async def _mock_complete(
        messages: list, tools: Any = None, system: str | None = None
    ) -> LLMResponse:
        nonlocal call_count
        call_count += 1
        return tool_response if call_count == 1 else final_response

    mock_llm_provider.complete = _mock_complete
    mock_llm_provider.stream_chunks = ["final ", "answer"]

    class _DispatcherWithResult:
        def get_tool_definitions(self) -> list:
            return []

        def dispatch(self, name: str, tool_input: dict) -> str:
            return "search result"

    runtime = AgentRuntime(mock_llm_provider, _DispatcherWithResult())

    events = [ev async for ev in runtime.execute_stream("test task")]

    tool_events = [ev for ev in events if ev["type"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["name"] == "search_knowledge_base"
    assert tool_events[0]["result"] == "search result"
    assert tool_events[0]["iteration"] == 0

    done = next(ev for ev in events if ev["type"] == "done")
    assert done["tool_calls"] == 1


@pytest.mark.anyio
async def test_agent_execute_stream_no_tool_calls(
    mock_llm_provider,
) -> None:
    """execute_stream() emits zero tool_call events when no tools are invoked."""
    from twin.agent.runtime import AgentRuntime

    runtime = AgentRuntime(mock_llm_provider, _NoOpDispatcher())

    events = [ev async for ev in runtime.execute_stream("test task")]

    tool_events = [ev for ev in events if ev["type"] == "tool_call"]
    assert tool_events == []

    done = next(ev for ev in events if ev["type"] == "done")
    assert done["tool_calls"] == 0


@pytest.mark.anyio
async def test_agent_execute_stream_event_order(
    mock_llm_provider,
) -> None:
    """Tool_call events always precede token events in the output."""
    from twin.agent.runtime import AgentRuntime

    call_count = 0
    tool_response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc_1", name="search_knowledge_base", input={"query": "q"})],
        stop_reason="tool_use",
    )

    async def _mock_complete(
        messages: list, tools: Any = None, system: str | None = None
    ) -> LLMResponse:
        nonlocal call_count
        call_count += 1
        return tool_response if call_count == 1 else LLMResponse(content="done")

    mock_llm_provider.complete = _mock_complete
    mock_llm_provider.stream_chunks = ["ans"]

    class _D:
        def get_tool_definitions(self) -> list:
            return []

        def dispatch(self, name: str, tool_input: dict) -> str:
            return "r"

    runtime = AgentRuntime(mock_llm_provider, _D())
    events = [ev async for ev in runtime.execute_stream("task")]

    types = [ev["type"] for ev in events]
    # All tool_call events must come before any token event
    first_token = next((i for i, t in enumerate(types) if t == "token"), len(types))
    last_tool = max((i for i, t in enumerate(types) if t == "tool_call"), default=-1)
    assert last_tool < first_token
