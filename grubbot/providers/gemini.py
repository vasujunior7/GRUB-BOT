import os
from litellm import completion
from .base import BaseProvider

class GeminiProvider(BaseProvider):
    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = completion(
            model=self.model,
            messages=messages,
            api_key=os.getenv("GEMINI_API_KEY")
        )
        return response.choices[0].message.content
