import google.generativeai as genai
import openai
import os
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def analyze_sentiment(text, provider="gemini"):
    """Analyzes sentiment using the selected provider (Gemini or OpenAI)"""
    if provider == "openai":
        return _openai_sentiment(text)
    elif provider == "gemini":
        return _gemini_sentiment(text)
    else:
        return "Invalid provider selected."

def _openai_sentiment(text):
    """Sentiment analysis using OpenAI"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Analyze the sentiment: {text}"}]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"OpenAI Error: {str(e)}"

def _gemini_sentiment(text):
    """Sentiment analysis using Gemini"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Analyze the sentiment: {text}")
    return response.text
