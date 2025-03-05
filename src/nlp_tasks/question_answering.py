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

def answer_question(question, context, provider="gemini"):
    """Answers a question based on the provided context"""
    if provider == "openai":
        return _openai_qa(question, context)
    elif provider == "gemini":
        return _gemini_qa(question, context)
    else:
        return "Invalid provider selected."

def _openai_qa(question, context):
    """Question Answering using OpenAI"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that answers questions based on the given context."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"OpenAI Error: {str(e)}"

def _gemini_qa(question, context):
    """Question Answering using Gemini"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Context: {context}\nQuestion: {question}")
    return response.text
