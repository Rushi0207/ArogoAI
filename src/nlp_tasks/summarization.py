import google.generativeai as genai
import openai
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def summarize_text(prompt, provider="gemini"):
    """Summarizes text using the selected provider (Gemini or OpenAI)"""
    if provider == "openai":
        return _openai_summarize(prompt)
    elif provider == "gemini":
        return _gemini_summarize(prompt)
    else:
        return "Invalid provider selected."

def _openai_summarize(prompt):
    """Summarization with OpenAI"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarize this: {prompt}"}]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"OpenAI Error: {str(e)}"

def _gemini_summarize(prompt):
    """Summarization with Gemini"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Summarize this: {prompt}")
    return response.text
