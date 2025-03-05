from src.models.openai_llm import OpenAILLM
from src.models.gemini_llm import GeminiLLM
from src.models.metaai_llm import MetaAILLM

class LLMWrapper:
    def __init__(self, provider="openai"):
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.llm = OpenAILLM()
        elif self.provider == "gemini":
            self.llm = GeminiLLM()
        elif self.provider == "metaai":
            self.llm = MetaAILLM()
        else:
            raise ValueError("Invalid LLM provider. Choose from: openai, gemini, metaai.")

    def generate_response(self, prompt):
        """Routes request to the appropriate LLM model."""
        return self.llm.generate_response(prompt)
