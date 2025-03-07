import os
import sys
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.evaluation.evaluator import ResponseEvaluator, SAMPLE_TEST_CASES
from src.rag.vector_store import VectorStore
from src.models.llm import LLMWrapper

def test_summary_evaluation():
    evaluator = ResponseEvaluator()
    result = evaluator.evaluate_summary(
        SAMPLE_TEST_CASES["summarization"][0]["generated"],
        SAMPLE_TEST_CASES["summarization"][0]["reference"]
    )
    assert 0 <= result["rouge-1"] <= 1

def test_code_evaluation():
    evaluator = ResponseEvaluator()
    test_case = SAMPLE_TEST_CASES["code_generation"][0]
    result = evaluator.evaluate_code(
        test_case["generated"], 
        test_case["test_cases"]
    )
    assert result["pass_rate"] == 1.0