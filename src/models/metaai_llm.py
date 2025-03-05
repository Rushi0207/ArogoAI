import os
from llama_cpp import Llama

META_MODEL_PATH = os.getenv("META_MODEL_PATH", "llama-2-7b-chat.ggmlv3.q4_0.bin")

class MetaAILLM:
    def __init__(self):
        self.llama = Llama(model_path=META_MODEL_PATH)

    def generate_response(self, prompt):
        """Generates a response using MetaAI's LLaMA models."""
        try:
            response = self.llama(prompt, max_tokens=512)
            return response["choices"][0]["text"]
        except Exception as e:
            return f" MetaAI Error: {str(e)}"
