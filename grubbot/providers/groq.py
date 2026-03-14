import os
from litellm import completion
from .base import BaseProvider

class GroqProvider(BaseProvider):
    def __init__(self, model: str = "groq/llama-3.3-70b-versatile"):
        self.model = model
        
    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = completion(
            model=self.model,
            messages=messages,
            api_key=os.getenv("GROQ_API_KEY")
        )
        return response.choices[0].message.content
