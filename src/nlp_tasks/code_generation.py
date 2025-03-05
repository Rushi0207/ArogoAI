import google.generativeai as genai
import openai
import transformers
import os
import requests  # For DeepSeek API
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com"

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def generate_code(prompt, provider="gemini"):
    """Generates code using the selected provider (Gemini, OpenAI, or DeepSeek)"""
    if provider == "openai":
        return _openai_code_generation(prompt)
    elif provider == "gemini":
        return _gemini_code_generation(prompt)
    elif provider == "metaai":
        return _metaai_code_generation(prompt)
    elif provider == "deepseek":
        return _deepseek_code_generation(prompt)
    else:
        return "Invalid provider selected."

def _openai_code_generation(prompt):
    """Generates code using OpenAI"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Write a code snippet for: {prompt}"}]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"OpenAI Error: {str(e)}"

def _gemini_code_generation(prompt):
    """Generates code using Gemini"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Write a code snippet for: {prompt}")
    return response.text

def _metaai_code_generation(prompt):
    """Generates code using MetaAI (LLaMA)"""
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/llama-7b-hf")
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/llama-7b-hf")
    encoded_input = tokenizer(prompt, return_tensors="pt")
    response = model.generate(**encoded_input)
    return tokenizer.decode(response[0], skip_special_tokens=True)

def _deepseek_code_generation(prompt):
    """Generates code using DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": f"Write a code snippet for: {prompt}"
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("generated_code", "No code generated in response.")
    except requests.exceptions.RequestException as e:
        return f"DeepSeek API Error: {str(e)}"