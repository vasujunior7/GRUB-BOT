from litellm import completion
from .base import BaseProvider

class OllamaProvider(BaseProvider):
    def __init__(self, model: str = "ollama/llama3"):
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # By default litellm will connect to localhost:11434 for ollama
        response = completion(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content
