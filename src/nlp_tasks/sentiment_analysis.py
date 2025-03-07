from src.models.llm_wrapper import LLMWrapper

def analyze_sentiment(text, provider):
    llm = LLMWrapper(provider)
    prompt = f"""
    Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral.
    
    Text: {text}

    Provide a classification and a brief explanation.

    If the text lacks clear sentiment, respond with:
    "The sentiment of the text is unclear or neutral."
    """
    return llm.generate_response(prompt)
