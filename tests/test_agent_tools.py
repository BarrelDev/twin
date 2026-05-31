"""Tests for agent tool definitions and dispatcher."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from twin.agent.tools import (
    ToolDefinition,
    get_kb_search_tool,
    search_knowledge_base,
    ToolDispatcher,
)
from twin.query.retriever import QueryResult
from twin.rag.context import FormattedContext


# ============================================================================
# ToolDefinition Tests
# ============================================================================


class TestToolDefinition:
    """Test the ToolDefinition dataclass."""

    def test_tool_definition_creation(self) -> None:
        """Test creating a ToolDefinition with required fields."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object", "properties": {}}

    def test_tool_definition_with_complex_schema(self) -> None:
        """Test ToolDefinition with a complex input schema."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }
        tool = ToolDefinition(
            name="search_tool",
            description="Search for things",
            input_schema=schema,
        )
        assert tool.input_schema == schema
        assert "query" in tool.input_schema["properties"]
        assert tool.input_schema["required"] == ["query"]


# ============================================================================
# get_kb_search_tool Tests
# ============================================================================


class TestGetKBSearchTool:
    """Test the get_kb_search_tool function."""

    def test_kb_search_tool_name(self) -> None:
        """Test that the KB search tool has the correct name."""
        tool = get_kb_search_tool()
        assert tool.name == "search_knowledge_base"

    def test_kb_search_tool_has_description(self) -> None:
        """Test that the KB search tool has a description."""
        tool = get_kb_search_tool()
        assert tool.description
        assert len(tool.description) > 0
        assert "knowledge base" in tool.description.lower()

    def test_kb_search_tool_input_schema_structure(self) -> None:
        """Test that the input schema is properly structured."""
        tool = get_kb_search_tool()
        schema = tool.input_schema

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_kb_search_tool_has_query_parameter(self) -> None:
        """Test that the query parameter is defined in the schema."""
        tool = get_kb_search_tool()
        properties = tool.input_schema["properties"]

        assert "query" in properties
        assert properties["query"]["type"] == "string"
        assert "description" in properties["query"]

    def test_kb_search_tool_has_optional_k_parameter(self) -> None:
        """Test that the k parameter is defined with a default."""
        tool = get_kb_search_tool()
        properties = tool.input_schema["properties"]

        assert "k" in properties
        assert properties["k"]["type"] == "integer"
        assert properties["k"]["default"] == 5

    def test_kb_search_tool_query_is_required(self) -> None:
        """Test that query is a required parameter."""
        tool = get_kb_search_tool()
        assert "query" in tool.input_schema["required"]

    def test_kb_search_tool_k_is_optional(self) -> None:
        """Test that k is not in the required list."""
        tool = get_kb_search_tool()
        assert "k" not in tool.input_schema["required"]


# ============================================================================
# search_knowledge_base Tests
# ============================================================================


class TestSearchKnowledgeBase:
    """Test the search_knowledge_base function."""

    def test_search_with_results(self) -> None:
        """Test search_knowledge_base when results are found."""
        mock_retriever = Mock()
        mock_chunk1 = Mock(
            chunk_id="chunk1",
            text="First result",
            source_path="file1.md",
        )
        mock_chunk2 = Mock(
            chunk_id="chunk2",
            text="Second result",
            source_path="file2.md",
        )
        mock_retriever.query.return_value = [mock_chunk1, mock_chunk2]

        mock_formatted_context = Mock()
        mock_formatted_context.text = "Formatted context with attribution"

        with patch(
            "twin.agent.tools.prepare_rag_context",
            return_value=mock_formatted_context,
        ):
            result = search_knowledge_base(mock_retriever, query="test query", k=5)

        assert isinstance(result, str)
        assert result == "Formatted context with attribution"
        mock_retriever.query.assert_called_once_with("test query", k=5)

    def test_search_with_no_results(self) -> None:
        """Test search_knowledge_base when no results are found."""
        mock_retriever = Mock()
        mock_retriever.query.return_value = []

        result = search_knowledge_base(mock_retriever, query="test query", k=5)

        assert isinstance(result, str)
        assert "No results found" in result
        assert "test query" in result

    def test_search_with_custom_k(self) -> None:
        """Test search_knowledge_base with custom k value."""
        mock_retriever = Mock()
        mock_retriever.query.return_value = []

        search_knowledge_base(mock_retriever, query="test", k=10)

        mock_retriever.query.assert_called_once_with("test", k=10)

    def test_search_with_default_k(self) -> None:
        """Test search_knowledge_base uses default k=5."""
        mock_retriever = Mock()
        mock_retriever.query.return_value = []

        search_knowledge_base(mock_retriever, query="test")

        mock_retriever.query.assert_called_once_with("test", k=5)

    def test_search_calls_prepare_rag_context(self) -> None:
        """Test that search_knowledge_base formats results with prepare_rag_context."""
        mock_retriever = Mock()
        mock_chunks = [Mock(), Mock()]
        mock_retriever.query.return_value = mock_chunks

        mock_formatted = Mock()
        mock_formatted.text = "Formatted output"

        with patch(
            "twin.agent.tools.prepare_rag_context",
            return_value=mock_formatted,
        ) as mock_prepare:
            search_knowledge_base(mock_retriever, query="test")

            mock_prepare.assert_called_once_with(mock_chunks)


# ============================================================================
# ToolDispatcher Tests
# ============================================================================


class TestToolDispatcher:
    """Test the ToolDispatcher class."""

    @pytest.fixture
    def mock_retriever(self) -> Mock:
        """Create a mock retriever for testing."""
        return Mock()

    @pytest.fixture
    def dispatcher(self, mock_retriever: Mock) -> ToolDispatcher:
        """Create a ToolDispatcher instance with a mock retriever."""
        return ToolDispatcher(mock_retriever)

    def test_dispatcher_initialization(self, mock_retriever: Mock) -> None:
        """Test ToolDispatcher initialization."""
        dispatcher = ToolDispatcher(mock_retriever)
        assert dispatcher.retriever == mock_retriever

    def test_get_tool_definitions(self, dispatcher: ToolDispatcher) -> None:
        """Test that get_tool_definitions returns a list."""
        tools = dispatcher.get_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_tool_definitions_contains_kb_search(
        self, dispatcher: ToolDispatcher
    ) -> None:
        """Test that available tools include search_knowledge_base."""
        tools = dispatcher.get_tool_definitions()
        tool_names = [tool.name for tool in tools]
        assert "search_knowledge_base" in tool_names

    def test_get_available_tool_names(self, dispatcher: ToolDispatcher) -> None:
        """Test get_available_tool_name returns tool names."""
        names = dispatcher.get_available_tool_name()
        assert isinstance(names, list)
        assert "search_knowledge_base" in names

    def test_get_available_tool_names_matches_definitions(
        self, dispatcher: ToolDispatcher
    ) -> None:
        """Test that available tool names match tool definitions."""
        definitions = dispatcher.get_tool_definitions()
        names = dispatcher.get_available_tool_name()

        definition_names = [tool.name for tool in definitions]
        assert set(names) == set(definition_names)

    def test_dispatch_search_knowledge_base(
        self, dispatcher: ToolDispatcher, mock_retriever: Mock
    ) -> None:
        """Test dispatching a search_knowledge_base tool call."""
        mock_retriever.query.return_value = []

        result = dispatcher.dispatch(
            "search_knowledge_base",
            {"query": "test query", "k": 5},
        )

        assert isinstance(result, str)
        mock_retriever.query.assert_called_once_with("test query", k=5)

    def test_dispatch_with_default_k(
        self, dispatcher: ToolDispatcher, mock_retriever: Mock
    ) -> None:
        """Test dispatch uses default k when not provided."""
        mock_retriever.query.return_value = []

        dispatcher.dispatch("search_knowledge_base", {"query": "test"})

        mock_retriever.query.assert_called_once_with("test", k=5)

    def test_dispatch_missing_query_parameter(
        self, dispatcher: ToolDispatcher
    ) -> None:
        """Test dispatch raises error when query parameter is missing."""
        with pytest.raises(ValueError, match="'query' parameter"):
            dispatcher.dispatch("search_knowledge_base", {"k": 5})

    def test_dispatch_unknown_tool(self, dispatcher: ToolDispatcher) -> None:
        """Test dispatch raises error for unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatcher.dispatch("nonexistent_tool", {})

    def test_dispatch_passes_through_result(
        self, dispatcher: ToolDispatcher, mock_retriever: Mock
    ) -> None:
        """Test that dispatch returns the tool's result."""
        expected_result = "Formatted search results"
        mock_retriever.query.return_value = []

        with patch(
            "twin.agent.tools.search_knowledge_base",
            return_value=expected_result,
        ):
            result = dispatcher.dispatch(
                "search_knowledge_base",
                {"query": "test"},
            )
            assert result == expected_result

    def test_dispatch_with_empty_query(
        self, dispatcher: ToolDispatcher, mock_retriever: Mock
    ) -> None:
        """Test dispatch with empty query string."""
        mock_retriever.query.return_value = []

        with pytest.raises(ValueError, match="'query' parameter"):
            dispatcher.dispatch("search_knowledge_base", {"query": ""})

    def test_dispatch_with_multiple_search_calls(
        self, dispatcher: ToolDispatcher, mock_retriever: Mock
    ) -> None:
        """Test multiple dispatched calls to search_knowledge_base."""
        mock_retriever.query.return_value = []

        dispatcher.dispatch("search_knowledge_base", {"query": "first"})
        dispatcher.dispatch("search_knowledge_base", {"query": "second"})

        assert mock_retriever.query.call_count == 2
        calls = mock_retriever.query.call_args_list
        assert calls[0][0][0] == "first"
        assert calls[1][0][0] == "second"
