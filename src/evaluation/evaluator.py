from rouge import Rouge
import ast
import pytest

class ResponseEvaluator:
    def __init__(self):
        self.rouge = Rouge()
    
    def evaluate_summary(self, generated, reference):
        scores = self.rouge.get_scores(generated, reference)
        return {"rouge-1": scores[0]["rouge-1"]["f"], "rouge-l": scores[0]["rouge-l"]["f"]}
    
    def evaluate_code(self, code_snippet, test_cases):
        try:
            exec(code_snippet)
            for case in test_cases:
                assert eval(case["assertion"]) == case["expected"]
            return {"pass_rate": 1.0}
        except Exception as e:
            return {"error": str(e), "pass_rate": 0.0}

    def evaluate_qa(self, answer, reference):
        return {"exact_match": int(answer.strip().lower() == reference.strip().lower())}

    def run_full_evaluation(self, test_cases):
        results = {}
        
        if "summarization" in test_cases:
            summ_results = []
            for case in test_cases["summarization"]:
                result = self.evaluate_summary(case["generated"], case["reference"])
                summ_results.append(result)
            results["summarization"] = summ_results
        
        if "code_generation" in test_cases:
            code_results = []
            for case in test_cases["code_generation"]:
                result = self.evaluate_code(case["generated"], case["test_cases"])
                code_results.append(result)
            results["code_generation"] = code_results

        if "question_answering" in test_cases:
            qa_results = []
            for case in test_cases["question_answering"]:
                result = self.evaluate_qa(case["generated"], case["reference"])
                qa_results.append(result)
            results["question_answering"] = qa_results

        return results


SAMPLE_TEST_CASES = {
    "summarization": [
        {
            "input": "A long article about climate change...",
            "generated": "Climate change impacts global temperatures.",
            "reference": "Global warming is caused by human activities."
        }
    ],
    "code_generation": [
        {
            "input": "Write Python code to add two numbers",
            "generated": "def add(a,b): return a+b",
            "test_cases": [
                {"assertion": "add(2,3)", "expected": 5},
                {"assertion": "add(-1,1)", "expected": 0}
            ]
        }
    ]
}