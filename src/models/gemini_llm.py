import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class GeminiLLM:
    @staticmethod
    def generate_response(prompt):
        """Generates a response using Google Gemini models."""
        if not GEMINI_API_KEY:
            return "Gemini API Key not found."

        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            return response.text if hasattr(response, "text") else "Gemini response error."
        except Exception as e:
            return f"Gemini API Error: {str(e)}"
