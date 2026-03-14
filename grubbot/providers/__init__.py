from .base import BaseProvider
from .gemini import GeminiProvider
from .groq import GroqProvider
from .ollama import OllamaProvider

def get_provider(name: str) -> BaseProvider:
    name_lower = name.lower()
    if name_lower == 'gemini':
        return GeminiProvider()
    elif name_lower == 'groq':
        return GroqProvider()
    elif name_lower == 'ollama':
        return OllamaProvider()
    elif name_lower.startswith('ollama/'):
        return OllamaProvider(model=name_lower)
    else:
        raise ValueError(f"Unknown provider: {name}")
