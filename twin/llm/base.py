from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self, messages: list[dict], tools: list[dict] | None, system: str
    ) -> dict:
        """
        Send a conversation to the model and return a response.

        Args:
            messages: Conversation history. Each item is a dict with 'role' and 'content'.
            tools: Optional list of tool definitions. Each item is a tool specification.
            system: System prompt string that guides model behavior.

        Returns:
            Response dict from the model, including the message content and metadata.
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