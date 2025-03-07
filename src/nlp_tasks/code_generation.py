from src.models.llm_wrapper import LLMWrapper

def generate_code(description, provider="openai"):
    llm = LLMWrapper(provider)
    prompt = f"""
    You are an expert software engineer. Generate optimized and well-commented code based on the following description:

    Description: {description}

    Follow best coding practices, use meaningful variable names, optimize for performance and readability, and include an example usage if relevant.

    If the description lacks details, respond with:
    "The request lacks enough details for precise code generation."
    """
    return llm.generate_response(prompt)
