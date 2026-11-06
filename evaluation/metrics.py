import re
from typing import List, Dict, Any

class AgriculturalEvaluator:
    """
    Quantitative Evaluation Metrics for Indian-language Agricultural RAG.
    Implements:
    - Groundedness / Faithfulness (RAG Triad)
    - Retrieval Context Recall@K
    - Answer Keyword Precision & Factual Overlap
    - Guardrail Safety Compliance
    """
    @staticmethod
    def compute_context_recall(expected_doc: str, retrieved_chunks: List[Dict[str, Any]], guardrail_status: str = "PASSED") -> float:
        if expected_doc == "out_of_domain" or guardrail_status == "BLOCKED":
            return 1.0
        retrieved_sources = [c.get("source", "") for c in retrieved_chunks]
        if expected_doc in retrieved_sources:
            return 1.0
        # Check partial stem match
        for s in retrieved_sources:
            if expected_doc.split(".")[0] in s:
                return 0.85
        return 0.0

    @staticmethod
    def compute_keyword_precision(expected_keywords: List[str], generated_answer: str) -> float:
        if not expected_keywords:
            return 1.0
        ans_lower = generated_answer.lower()
        matched = 0
        for kw in expected_keywords:
            kw_lower = kw.lower().strip()
            # If slash separated variants e.g. "4.0% / 4%"
            variants = [v.strip() for v in kw_lower.split("/")]
            if any(v in ans_lower for v in variants):
                matched += 1
            else:
                # Check individual significant tokens (>3 chars)
                tokens = [t for t in kw_lower.split() if len(t) > 3]
                if tokens and any(t in ans_lower for t in tokens):
                    matched += 0.7
        return round(min(1.0, matched / len(expected_keywords)), 3)

    @staticmethod
    def compute_groundedness(generated_answer: str, retrieved_chunks: List[Dict[str, Any]], guardrail_status: str) -> float:
        """
        Evaluates whether facts in generated_answer are grounded in retrieved context or safely intercepted.
        """
        if guardrail_status == "BLOCKED":
            return 1.0

        if not retrieved_chunks:
            return 0.0

        combined_context = " ".join([c.get("text", "") for c in retrieved_chunks]).lower()
        
        # Check if chemical entities / numbers in answer are present in context
        numbers_in_answer = re.findall(r'\b\d+(?:\.\d+)?\b', generated_answer)
        if not numbers_in_answer:
            return 0.85

        supported_nums = sum(1 for num in numbers_in_answer if num in combined_context)
        ratio = supported_nums / len(numbers_in_answer)
        # Scaled faithfulness
        return round(min(1.0, 0.5 + 0.5 * ratio), 3)

    @staticmethod
    def compute_overall_score(recall: float, kw_precision: float, groundedness: float) -> float:
        # Weighted evaluation composite score
        score = 0.35 * recall + 0.35 * kw_precision + 0.30 * groundedness
        return round(score, 3)
