"""Tool definitions and dispatch for the agent runtime.

This module defines the tools available to the agent (e.g., KB search) and
provides a dispatcher to route tool calls from the LLM to their implementations.
"""

from typing import Any

from twin.query.retriever import Retriever
from twin.rag.context import prepare_rag_context


class ToolDefinition:
    """
    Schema for a tool that the LLM can call.

    Represents a tool definition in the format expected by the LLM provider
    (e.g., Anthropic's tool format).
    """

    # TODO: Define the structure (name, description, input_schema)
    pass


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
    pass


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
    # TODO: Implement - follow these steps:
    # 1. Call retriever.query(query, k=k)
    # 2. Call prepare_rag_context(chunks) to format with sources
    # 3. Return formatted.text (the context string)
    pass


class ToolDispatcher:
    """Routes tool calls from the LLM to their implementations."""

    def __init__(self, retriever: Retriever) -> None:
        """
        Initialize the tool dispatcher.

        Args:
            retriever: Retriever instance for KB search operations.
        """
        # TODO: Store retriever for use in dispatch
        pass

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
        # TODO: Implement - route based on tool_name:
        # - "search_knowledge_base": call search_knowledge_base()
        # - Unknown: raise ValueError(f"Unknown tool: {tool_name}")
        pass

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """
        Return all available tools.

        Returns:
            List of ToolDefinition objects describing available tools.
            Passed to the LLM to tell it what it can call.
        """
        # TODO: Implement - return a list containing:
        # - get_kb_search_tool()
        # - Any other tools added later
        pass
