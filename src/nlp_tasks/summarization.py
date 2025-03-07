from src.models.llm_wrapper import LLMWrapper

def summarize_text(text, provider="openai"):
    if len(text.split()) < 10:
        return f"Summary: {text}"  

    llm = LLMWrapper(provider)
    prompt = f"""
    Summarize the following text concisely while keeping key details:
    
    Text: {text}

    Ensure readability and clarity. Keep sentences short and professional.
    
    If the text is too short or lacks meaningful content, respond with:
    "The provided text is too short for summarization."
    """
    return llm.generate_response(prompt)
