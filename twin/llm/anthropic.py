import os
from anthropic import Anthropic
from .base import LLMProvider

_MAX_TOKENS = 1024


class Claude(LLMProvider):
    """Anthropic Claude provider for the LLM client."""

    def __init__(self, model: str | None = None) -> None:
        """
        Initialize the Claude provider.

        Reads ANTHROPIC_API_KEY from environment. Raises ValueError if missing.
        Fetches available models from the Anthropic API at initialization.

        Args:
            model: Model identifier to use. If None, defaults to the first available model.

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not set or no models are available.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Please set it before using the Claude provider."
            )

        self.client = Anthropic(api_key=api_key)

        # Fetch available models from the API
        models_response = self.client.models.list()
        self._available_models = [model.id for model in models_response.data]

        if not self._available_models:
            raise ValueError("No models available from Anthropic API.")

        # Use provided model or default to the first available
        if model:
            if model not in self._available_models:
                raise ValueError(
                    f"Model '{model}' not available. "
                    f"Available models: {', '.join(self._available_models)}"
                )
            self.model = model
        else:
            self.model = self._available_models[0]

    def complete(
        self, messages: list[dict], tools: list[dict] | None, system: str
    ) -> dict:
        """
        Send a conversation to Claude and return the response.

        Args:
            messages: Conversation history with 'role' and 'content' fields.
            tools: Optional tool definitions to make available to the model.
            system: System prompt to guide model behavior.

        Returns:
            Response dict with 'content', 'stop_reason', and other metadata from the API.
        """
        kwargs = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "system": system,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        return self.client.messages.create(**kwargs)

    def list_models(self) -> list[str]:
        """
        Return the list of available Claude models.

        Returns:
            List of model identifiers available through this provider.
        """
        return self._available_models