import os
import openai
import google.generativeai as genai

# Load API Keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)


def review_code(code_snippet, provider=openai):
    prompt = f"""
    You are an expert Python code reviewer. Analyze the following Python code and provide detailed feedback.

    **Your Review Should Cover:**
    1. **Syntax & Logic Errors:** Identify any issues in the code.
    2. **Best Practices:** Ensure adherence to Pythonic conventions (PEP-8).
    3. **Performance Improvements:** Suggest optimizations.
    4. **Readability & Maintainability:** Recommend improvements.
    5. **Security Concerns:** Highlight potential vulnerabilities.
    6. **Example Fixes:** Provide corrected versions if necessary.

    **Python Code to Review:**
    ```python
    {code_snippet}
    ```

    **Output Format:**
    - **Issues Found:** (list of problems)
    - **Suggested Fixes:** (code improvements with explanations)
    - **Best Practices:** (additional recommendations)
    """

    if provider.lower() == "openai":
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Python code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI API Error: {str(e)}"

    elif provider.lower() == "gemini":
        if not GEMINI_API_KEY:
            return "Gemini API Key not found. Please set it in your environment variables."
        
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text if hasattr(response, "text") else "Gemini response error."
        except Exception as e:
            return f"Gemini API Error: {str(e)}"

    return "Invalid AI provider selected. Please choose 'openai' or 'gemini'."
