import os
from typing import Any
from anthropic import Anthropic
from .base import LLMProvider

_MAX_TOKENS = 1024


class Claude(LLMProvider):
    """Anthropic Claude provider for the LLM client."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        """
        Initialize the Claude provider.

        Args:
            model: Model identifier to use. If None, defaults to the first available model.
            api_key: Anthropic API key. If None, falls back to ANTHROPIC_API_KEY env var.

        Raises:
            ValueError: If no API key is found or no models are available.
        """
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No Anthropic API key found.\n"
                "Run: twin config set-key\n"
                "Or set ANTHROPIC_API_KEY in your environment."
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
        self, messages: list[dict[str, Any]], tools: list[dict] | None, system: str
    ) -> Any:
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

    def extract_answer(self, response: Any) -> str:
        """
        Extract the answer text from an Anthropic Message response.

        The Anthropic API returns a Message object with a content list containing
        ContentBlock objects. This method extracts the text from the first content block.

        Args:
            response: Message response object from self.complete().

        Returns:
            The answer text from the first content block.

        Raises:
            ValueError: If response has no content or first block is not text.
        """
        if not hasattr(response, "content") or not response.content:
            raise ValueError("Response has no content blocks")

        first_block = response.content[0]
        if not hasattr(first_block, "text"):
            raise ValueError(
                f"First content block is not text (type: {type(first_block).__name__})"
            )

        return first_block.text

    def list_models(self) -> list[str]:
        """
        Return the list of available Claude models.

        Returns:
            List of model identifiers available through this provider.
        """
        return self._available_models