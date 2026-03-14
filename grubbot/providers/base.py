from abc import ABC, abstractmethod

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response using the given prompt and system instructions."""
        pass
