"""Abstract LLM provider interface and shared response/tool types.

All provider methods that communicate with LLM APIs are async.
Use asyncio.run() at the CLI boundary to execute them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from twin.config import ModelInfo


@dataclass
class ToolDefinition:
    """Schema for a tool the LLM can invoke."""

    name: str
    """Unique tool name, e.g. 'search_knowledge_base'."""

    description: str
    """Human-readable description of what the tool does."""

    input_schema: dict
    """JSON Schema for tool input parameters."""


@dataclass
class ToolCall:
    """A single tool call emitted by the LLM."""

    id: str
    """Provider-assigned call identifier used to match results back."""

    name: str
    """Name of the tool that was called."""

    input: dict[str, Any]
    """Arguments the model supplied for this call."""


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str | None
    """Text content of the response, or None when the response is tool-only."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    """Tool calls the model wants to make (empty for final answers)."""

    stop_reason: str = "end_turn"
    """Why the model stopped: 'end_turn', 'tool_use', 'max_tokens', etc."""

    prompt_tokens: int = 0
    """Number of input tokens consumed."""

    completion_tokens: int = 0
    """Number of output tokens generated."""


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Providers convert their native API responses into LLMResponse and accept
    messages in Anthropic-style format (content arrays with typed blocks).
    Non-Anthropic providers perform the format conversion internally in
    complete() so the rest of the application stays provider-agnostic.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send a conversation to the model and return a normalized response.

        Messages use Anthropic-style content blocks:
        - Tool calls: assistant role, content list with tool_use blocks
        - Tool results: user role, content list with tool_result blocks

        Args:
            messages: Conversation history.
            tools: Tool definitions the model may call.
            system: System prompt guiding model behavior.

        Returns:
            Normalized LLMResponse with content, tool_calls, and token usage.
        """

    def extract_answer(self, response: LLMResponse) -> str:
        """
        Extract the text answer from a normalized LLM response.

        Args:
            response: LLMResponse from complete().

        Returns:
            The text content, or an empty string if the response has no text.
        """
        return response.content or ""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response token-by-token.

        Args:
            messages: Conversation history.
            tools: Tool definitions the model may call.
            system: System prompt.

        Yields:
            Text chunks as they arrive from the provider.
        """

    @abstractmethod
    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        """
        Estimate the cost in USD for a given token count.

        Args:
            prompt_tokens: Input token count.
            completion_tokens: Output token count.

        Returns:
            Estimated cost in USD, or None for providers with no public pricing.
        """

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """
        Return available models for this provider.

        Returns:
            List of ModelInfo objects for each available model.
        """
