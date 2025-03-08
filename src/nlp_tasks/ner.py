from src.models.llm_wrapper import LLMWrapper

def extract_entities(text, provider="openai"):
    llm = LLMWrapper(provider)
    prompt = f"""
    Identify named entities in the following text and classify them as:
    
    - Person
    - Organization
    - Location
    - Date/Time
    - Event
    - Product
    - Law/Regulation
    - Work of Art (Book, Movie, Song, etc.)
    - Medical Term
    - Currency (Money)
    - Other (if none of the above apply)

    Text: "{text}"

    Format the response as a well-structured **text description** like this:

    "Named Entities Found:
    - Elon Musk (Person)
    - Tesla (Organization)
    - Louvre (Location)
    - Mona Lisa (Work of Art)
    - GDPR (Law/Regulation)

    If **no named entities** are detected, return:
    "No named entities detected in the given text."
    """
    
    return llm.generate_response(prompt)
