from abc import ABC, abstractmethod

class LLMProvider(ABC):
    
    @abstractmethod
    def complete(messages, tools, system):
        pass

    @abstractmethod
    def list_models():
        pass