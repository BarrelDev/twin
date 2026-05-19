from base import LLMProvider

from anthropic import Anthropic
import os

MAX_TOKENS = 1024

class Claude(LLMProvider):
    def __init__(self):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, messages, tools, system):
        return self.client.messages.create(
            max_tokens=MAX_TOKENS,
            messages=messages,
            model=system
        )
    
    def list_models(self):
        return