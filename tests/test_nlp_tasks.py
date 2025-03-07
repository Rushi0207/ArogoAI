import os
import sys
import pytest
import json
from unittest.mock import patch, MagicMock

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.llm_wrapper import LLMWrapper
from src.nlp_tasks.summarization import summarize_text
from src.nlp_tasks.sentiment_analysis import analyze_sentiment
from src.nlp_tasks.ner import extract_entities
from src.nlp_tasks.question_answering import answer_question
from src.nlp_tasks.code_generation import generate_code
from src.nlp_tasks.code_review import review_code
from src.evaluation.evaluator import ResponseEvaluator
from src.utils.cache_manager import CacheManager
from src.rag.vector_store import VectorStore


PROVIDER = "gemini"

def test_summarization():
    text = "Artificial intelligence is transforming the world."
    summary = summarize_text(text, provider=PROVIDER)
    assert isinstance(summary, str) and len(summary) > 0

def test_sentiment_analysis():
    text = "I love this product!"
    sentiment = analyze_sentiment(text, provider=PROVIDER)
    assert "Positive" in sentiment or "Negative" in sentiment or "Neutral" in sentiment

def test_ner():
    text = "Elon Musk founded SpaceX."
    entities = extract_entities(text, provider=PROVIDER)
    assert "Elon Musk" in entities and "Person" in entities

def test_question_answering():
    context = "Paris is the capital of France."
    question = "What is the capital of France?"
    answer = answer_question(question, context, provider=PROVIDER)
    assert "Paris" in answer

def test_code_generation():
    prompt = "Write a Python function to add two numbers."
    code = generate_code(prompt, provider=PROVIDER)
    assert "def" in code and "return" in code

def test_code_review():
    code = "def add(a, b): return a + b"
    review = review_code(code, provider=PROVIDER)
    assert isinstance(review, str) and len(review) > 0

def test_cache_manager():
    cache = CacheManager()
    prompt_hash = cache._hash_prompt("Hello", PROVIDER)
    cache.set_cached_response(prompt_hash, "Hello Response")
    assert cache.get_cached_response(prompt_hash) == "Hello Response"

def test_vector_store():
    store = VectorStore()
    store.add_document("Artificial Intelligence is the future.")
    results = store.search("AI", top_k=1)
    assert len(results) > 0

def test_response_evaluator():
    evaluator = ResponseEvaluator()
    result = evaluator.evaluate_summary("AI is great", "Artificial Intelligence is great")
    assert "rouge-1" in result and "rouge-l" in result
