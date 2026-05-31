from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict] | None, system: str
    ) -> Any:
        """
        Send a conversation to the model and return a response.

        Args:
            messages: Conversation history. Each item is a dict with 'role' and 'content'.
                      Content may be a plain string or a list of content-block dicts.
            tools: Optional list of tool definitions. Each item is a tool specification.
            system: System prompt string that guides model behavior.

        Returns:
            Opaque provider response object. Pass to extract_answer() to get text,
            or inspect directly for tool-use blocks in provider-specific code.
        """
        pass

    @abstractmethod
    def extract_answer(self, response: Any) -> str:
        """
        Extract the answer text from an LLM response.

        This method abstracts away provider-specific response formats, allowing the
        RAG pipeline and other consumers to be provider-agnostic.

        Args:
            response: Response object from complete().

        Returns:
            The answer text extracted from the response.

        Raises:
            ValueError: If the response format is unexpected or no text can be extracted.
        """
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """
        Return the available models for this provider.

        Returns:
            List of model identifiers available for this provider.
        """
        pass