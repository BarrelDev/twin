"""Tool definitions and dispatch for the agent runtime.

This module defines the tools available to the agent (e.g., KB search) and
provides a dispatcher to route tool calls from the LLM to their implementations.

ToolDefinition lives in twin.llm.base (it's an LLM-layer concept); it is
re-exported here for the convenience of callers that import from this module.
"""

from typing import Any

from twin.llm.base import ToolDefinition  # re-exported for backward compat
from twin.query.retriever import Retriever
from twin.rag.context import prepare_rag_context

__all__ = ["ToolDefinition", "get_kb_search_tool", "search_knowledge_base", "ToolDispatcher"]


def get_kb_search_tool() -> ToolDefinition:
    """
    Define the knowledge base search tool.

    Returns the tool definition that tells the LLM how to call KB search,
    including the parameter schema and description.

    Returns:
        ToolDefinition for KB search.
    """
    # TODO: Implement - return a ToolDefinition with:
    # - name: "search_knowledge_base"
    # - description: Clear description of what it does
    # - input_schema: JSON schema for the query parameter
    return ToolDefinition(
        name="search_knowledge_base",
        description=(
            "Search the knowledge base for chunks relevant to a query. "
            "Returns up to k formatted chunks with source attribution. "
            "Use this to answer questions or find context about topics in your knowledge base."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g., 'How do decorators work?')"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    )


def search_knowledge_base(retriever: Retriever, query: str, k: int = 5) -> str:
    """
    Search the knowledge base and return formatted results.

    Retrieves the top-k chunks matching the query and formats them with source
    attribution so the LLM can reason over them.

    Args:
        retriever: Retriever instance for KB search.
        query: The search query string.
        k: Number of chunks to retrieve (default: 5).

    Returns:
        Formatted context string with source attribution, ready for LLM processing.
    """
    queries = retriever.query(query, k=k)
    if not queries:
        return f"No results found for query: '{query}'"
    
    fmt_context = prepare_rag_context(queries)
    return fmt_context.text


class ToolDispatcher:
    """Routes tool calls from the LLM to their implementations."""

    def __init__(self, retriever: Retriever) -> None:
        """
        Initialize the tool dispatcher.

        Args:
            retriever: Retriever instance for KB search operations.
        """
        self.retriever = retriever

    def dispatch(self, tool_name: str, tool_input: dict) -> str:
        """
        Route a tool call to its implementation.

        Args:
            tool_name: Name of the tool to call (e.g., "search_knowledge_base").
            tool_input: Dict with tool parameters (e.g., {"query": "...", "k": 5}).

        Returns:
            Tool result as a string (formatted for LLM processing).

        Raises:
            ValueError: If tool_name is not recognized.
        """
        match tool_name:
            case "search_knowledge_base":
                query = tool_input.get("query")
                if not query:
                    raise ValueError("search_knowledge_base requires 'query' parameter")
                k = tool_input.get("k", 5)
                return search_knowledge_base(self.retriever, query=query, k=k)
            case _:
                raise ValueError(f"Unknown tool: {tool_name}") 

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """
        Return all available tools.

        Returns:
            List of ToolDefinition objects describing available tools.
            Passed to the LLM to tell it what it can call.
        """
        return [get_kb_search_tool()]

    def get_available_tool_name(self) -> list[str]:
        """Return names of all the available tools"""
        return [tool.name for tool in self.get_tool_definitions()]
