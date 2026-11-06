import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TEST_SUITE_PATH, EVAL_RESULTS_PATH
from src.agents.farmer_agent import KrishiFarmerAgent
from evaluation.metrics import AgriculturalEvaluator

def run_benchmark():
    print("=" * 60)
    print("🌾 KrishiMitra AI: Running Benchmark Evaluation on 45 Test Cases")
    print("=" * 60)

    if not TEST_SUITE_PATH.exists():
        print(f"Error: Test suite not found at {TEST_SUITE_PATH}")
        return

    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    agent = KrishiFarmerAgent()
    evaluator = AgriculturalEvaluator()

    results = []
    category_scores = {}
    language_scores = {}

    start_total_time = time.time()

    for idx, tc in enumerate(test_cases, 1):
        t0 = time.time()
        tc_id = tc["id"]
        lang = tc["language"]
        cat = tc["category"]
        query = tc["query"]
        ground_truth = tc["ground_truth"]
        expected_doc = tc.get("source_doc", "")
        expected_keywords = tc.get("expected_keywords", [])

        # Query Agent
        response_text, meta = agent.ask(query, lang=lang)
        latency = round((time.time() - t0) * 1000, 2)  # ms

        # Compute Metrics
        retrieved_chunks = meta.get("retrieved_chunks", [])
        guardrail_status = meta.get("guardrail_status", "PASSED")

        context_recall = evaluator.compute_context_recall(expected_doc, retrieved_chunks, guardrail_status)
        kw_precision = evaluator.compute_keyword_precision(expected_keywords, response_text)
        groundedness = evaluator.compute_groundedness(response_text, retrieved_chunks, guardrail_status)
        overall_score = evaluator.compute_overall_score(context_recall, kw_precision, groundedness)

        item_result = {
            "id": tc_id,
            "category": cat,
            "language": lang,
            "language_name": tc.get("language_name", lang),
            "difficulty": tc.get("difficulty", "medium"),
            "query": query,
            "expected_doc": expected_doc,
            "top_retrieved": meta.get("top_citation"),
            "guardrail_status": guardrail_status,
            "context_recall": context_recall,
            "keyword_precision": kw_precision,
            "groundedness": groundedness,
            "overall_score": overall_score,
            "latency_ms": latency,
            "generated_response": response_text[:200] + "..."
        }
        results.append(item_result)

        # Track Category stats
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(overall_score)

        # Track Language stats
        if lang not in language_scores:
            language_scores[lang] = []
        language_scores[lang].append(overall_score)

        print(f"[{idx:02d}/{len(test_cases)}] {tc_id} ({lang} | {cat}): Score = {overall_score:.2f} (Recall: {context_recall}, KW: {kw_precision}, Grounded: {groundedness}) in {latency}ms")

    total_duration = round(time.time() - start_total_time, 2)

    avg_recall = round(sum(r["context_recall"] for r in results) / len(results), 3)
    avg_kw_precision = round(sum(r["keyword_precision"] for r in results) / len(results), 3)
    avg_groundedness = round(sum(r["groundedness"] for r in results) / len(results), 3)
    avg_overall = round(sum(r["overall_score"] for r in results) / len(results), 3)
    avg_latency = round(sum(r["latency_ms"] for r in results) / len(results), 2)

    cat_summary = {c: round(sum(scores) / len(scores), 3) for c, scores in category_scores.items()}
    lang_summary = {l: round(sum(scores) / len(scores), 3) for l, scores in language_scores.items()}

    summary_data = {
        "benchmark_metadata": {
            "total_test_cases": len(results),
            "total_evaluation_time_seconds": total_duration,
            "average_latency_ms": avg_latency,
            "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "aggregate_metrics": {
            "mean_context_recall": avg_recall,
            "mean_keyword_precision": avg_kw_precision,
            "mean_groundedness": avg_groundedness,
            "mean_overall_score": avg_overall
        },
        "category_breakdown": cat_summary,
        "language_breakdown": lang_summary,
        "detailed_results": results
    }

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("🎯 BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"• Total Test Cases: {len(results)}")
    print(f"• Mean Context Recall: {avg_recall * 100:.1f}%")
    print(f"• Mean Keyword Precision: {avg_kw_precision * 100:.1f}%")
    print(f"• Mean Groundedness / Faithfulness: {avg_groundedness * 100:.1f}%")
    print(f"• Mean Overall Benchmark Score: {avg_overall * 100:.1f}%")
    print(f"• Average Latency: {avg_latency} ms")
    print(f"• Detailed results saved to: {EVAL_RESULTS_PATH}")
    print("=" * 60)

    return summary_data

if __name__ == "__main__":
    run_benchmark()
