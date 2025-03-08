from src.models.llm_wrapper import LLMWrapper

def answer_question(question, context, provider="openai"):
    llm = LLMWrapper(provider)
    prompt = f"""
    Answer the following question **ONLY** using the provided context.
    
    Context: {context}
    Question: {question}

    If the question cannot be answered using the context, respond with:
    "The question is out of context."
    """
    return llm.generate_response(prompt)
