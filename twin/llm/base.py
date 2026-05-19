from abc import ABC, abstractmethod

class LLMProvider(ABC):
    
    @abstractmethod
    def complete(self, messages, tools, system):
        pass

    @abstractmethod
    def list_models(self):
        pass