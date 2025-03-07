from src.models.llm_wrapper import LLMWrapper

def review_code(code_snippet, provider="openai"):
    llm = LLMWrapper(provider)
    prompt = f"""
    You are a senior software engineer reviewing the following code:

    Code:
    ```
    {code_snippet}
    ```

    Identify potential bugs, security vulnerabilities, inefficiencies, and suggest improvements for readability, maintainability, and performance.

    If the input is not valid code, respond with:
    "The provided input is not valid code."
    """
    return llm.generate_response(prompt)
