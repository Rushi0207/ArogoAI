import openai
import os
import google.generativeai as genai
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

#Envirement variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

class LLMWrapper:
    def __init__(self, provider="openai"):
        self.provider = provider
        
        if provider == "openai":
            self.api_key = OPENAI_API_KEY
        elif provider == "huggingface":
            self.model = pipeline("text-generation", model="facebook/opt-1.3b")

    def generate_response(self, prompt):
        if self.provider == "openai":
            return self._openai_response(prompt)
        elif self.provider == "gemini":
            return self._gemini_response(prompt)
        else:
            return "Invalid provider selected."

    def _openai_response(self, prompt):
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content

    def _gemini_response(self, prompt):
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text