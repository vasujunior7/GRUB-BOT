import time
from abc import ABC, abstractmethod
from loguru import logger

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response using the given prompt and system instructions."""
        pass
        
    def generate_with_retry(self, prompt: str, system: str = "") -> str:
        """Wrapper for generate with exponential backoff."""
        max_retries = 5
        base_wait = 2
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, system)
            except Exception as e:
                wait_time = base_wait * (2 ** attempt)
                logger.warning(f"LLM generation failed: {e}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
        raise RuntimeError(f"Failed to generate after {max_retries} attempts.")
