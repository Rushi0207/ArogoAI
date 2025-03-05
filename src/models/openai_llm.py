import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OpenAILLM:
    @staticmethod
    def generate_response(prompt):
        """Generates a response using OpenAI's GPT models."""
        if not OPENAI_API_KEY:
            return "OpenAI API Key not found."

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI API Error: {str(e)}"
