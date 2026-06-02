"""Tool definitions and dispatch for the agent runtime.

This module defines the tools available to the agent (e.g., KB search) and
provides a dispatcher to route tool calls from the LLM to their implementations.

ToolDefinition lives in twin.llm.base (it's an LLM-layer concept); it is
re-exported here for the convenience of callers that import from this module.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from twin.llm.base import ToolDefinition  # re-exported for backward compat
from twin.query.retriever import Retriever
from twin.rag.context import prepare_rag_context

__all__ = [
    "ToolDefinition",
    "VaultWriter",
    "get_kb_search_tool",
    "get_write_vault_note_tool",
    "search_knowledge_base",
    "ToolDispatcher",
]

# Characters that are illegal or unsafe in filenames across platforms
UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class VaultWriter:
    """Writes agent-generated notes to the Obsidian vault's Agents/ folder."""

    def __init__(self, vault_path: Path) -> None:
        """
        Initialize the vault writer.

        Args:
            vault_path: Root path of the Obsidian vault.
        """
        self._vault = vault_path
        self._agents_dir = vault_path / "Agents"
        self._agents_dir.mkdir(parents=True, exist_ok=True)

    def write_vault_note(
        self, title: str, content: str, tags: list[str] | None = None
    ) -> Path:
        """
        Write an agent-generated note to <vault>/Agents/<task-slug>/<timestamp>-<title>.md.

        Never writes outside the Agents/ directory. This is enforced at path level.

        Args:
            title: Note title. Used as filename (sanitized) and H1 heading.
            content: Markdown body content.
            tags: Optional list of Obsidian tags.

        Returns:
            Path to the created file, relative to vault root.

        Raises:
            ValueError: If the sanitized path would escape the Agents/ directory.
        """
        safe_title = UNSAFE_PATH_CHARS.sub("_", title)[:80].strip("._") or "note"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        note_dir = self._agents_dir / safe_title
        note_path = note_dir / f"{timestamp}-{safe_title}.md"

        # HARD BOUNDARY CHECK — never allow escaping Agents/
        try:
            note_path.resolve().relative_to(self._agents_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Attempted write outside Agents/ directory: {note_path}"
            )

        note_dir.mkdir(parents=True, exist_ok=True)

        all_tags = (tags or []) + ["twin-generated"]
        now_iso = datetime.now(timezone.utc).isoformat()
        frontmatter = (
            f"---\n"
            f"generated_by: twin-agent\n"
            f"task: {title}\n"
            f"created: {now_iso}\n"
            f"tags: [{', '.join(all_tags)}]\n"
            f"---\n\n"
        )
        note_path.write_text(
            frontmatter + f"# {title}\n\n" + content, encoding="utf-8"
        )
        return note_path.relative_to(self._vault)


def get_write_vault_note_tool() -> ToolDefinition:
    """
    Define the write_vault_note tool.

    Returns:
        ToolDefinition for writing notes to the Obsidian vault.
    """
    return ToolDefinition(
        name="write_vault_note",
        description=(
            "Write an agent-generated note to the Obsidian vault's Agents/ folder. "
            "Use this to save findings, summaries, or any content worth keeping in the vault."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Note title (used as filename and H1 heading)",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown body content",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of Obsidian tags to apply",
                },
            },
            "required": ["title", "content"],
        },
    )


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

    def __init__(
        self, retriever: Retriever, vault_writer: VaultWriter | None = None
    ) -> None:
        """
        Initialize the tool dispatcher.

        Args:
            retriever: Retriever instance for KB search operations.
            vault_writer: Optional VaultWriter for note write-back. Only provided
                when a vault path is configured in ConfigManager.
        """
        self.retriever = retriever
        self._vault_writer = vault_writer

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
            case "write_vault_note":
                if not self._vault_writer:
                    return (
                        "Error: No vault path configured. "
                        "Run: twin config set-vault <path>"
                    )
                title = tool_input.get("title", "")
                content = tool_input.get("content", "")
                tags = tool_input.get("tags")
                rel_path = self._vault_writer.write_vault_note(title, content, tags)
                return f"Note written to: {rel_path}"
            case _:
                raise ValueError(f"Unknown tool: {tool_name}")

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """
        Return all available tools.

        The write_vault_note tool is only included when a vault writer is configured.

        Returns:
            List of ToolDefinition objects describing available tools.
            Passed to the LLM to tell it what it can call.
        """
        tools: list[ToolDefinition] = [get_kb_search_tool()]
        if self._vault_writer:
            tools.append(get_write_vault_note_tool())
        return tools

    def get_available_tool_name(self) -> list[str]:
        """Return names of all available tools."""
        return [tool.name for tool in self.get_tool_definitions()]
