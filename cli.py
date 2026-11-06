"""
KrishiMitra AI - Command Line Interface (CLI)
Allows rapid testing, grading, and querying without launching the web server.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.farmer_agent import KrishiFarmerAgent
from src.agents.tools.npk_calculator import NPKFertilizerCalculator
from src.agents.tools.weather_tool import AgroWeatherAdvisor
from src.agents.tools.scheme_matcher import GovernmentSchemeMatcher

def main():
    parser = argparse.ArgumentParser(description="KrishiMitra AI: Command Line Advisory Runner")
    parser.add_argument("--query", "-q", type=str, help="Agricultural query text")
    parser.add_argument("--lang", "-l", type=str, default="hi", choices=["hi", "en", "te", "ta", "mr", "bn"], help="Language code")
    parser.add_argument("--calc-npk", type=str, choices=["rice", "wheat", "cotton", "sugarcane", "maize", "mustard", "tomato"], help="Calculate NPK fertilizer for crop")
    parser.add_argument("--acres", type=float, default=1.0, help="Land acreage for NPK calculation")
    parser.add_argument("--soil", type=str, default="medium", choices=["low", "medium", "high"], help="Soil fertility status")
    parser.add_argument("--weather", action="store_true", help="Run weather spray test")
    parser.add_argument("--benchmark", action="store_true", help="Run full 45 test case benchmark")

    args = parser.parse_args()

    if args.benchmark:
        from evaluation.run_evaluation import run_benchmark
        run_benchmark()
        return

    if args.calc_npk:
        calc = NPKFertilizerCalculator()
        res = calc.calculate(args.calc_npk, acres=args.acres, soil_fertility=args.soil)
        print("\n" + "=" * 60)
        print(f"🌾 Fertilizer Prescription for {res['crop_name']} ({args.acres} Acres, {args.soil.upper()} Soil)")
        print("=" * 60)
        print(f"• Urea (46% N): {res['fertilizer_recommendation_kg']['Urea (46% N)']} kg ({res['commercial_bags_50kg']['Urea Bags']} Bags)")
        print(f"• DAP (18:46:0): {res['fertilizer_recommendation_kg']['DAP (18:46:0)']} kg ({res['commercial_bags_50kg']['DAP Bags']} Bags)")
        print(f"• MOP (60% K2O): {res['fertilizer_recommendation_kg']['MOP (60% K2O)']} kg ({res['commercial_bags_50kg']['MOP Bags']} Bags)")
        print(f"\nTimeline:\n{res['application_schedule']}")
        print(f"Micronutrient: {res['micronutrient_tip']}\n")
        return

    if args.weather:
        weather = AgroWeatherAdvisor()
        res = weather.evaluate_spray_conditions(temp_c=26.0, humidity_pct=65.0, wind_kmh=8.0, rain_prob_pct=15.0)
        print("\n" + "=" * 60)
        print(f"🌦️ Weather Spray Verdict: {res['spray_verdict']}")
        print("=" * 60)
        for w in res["advisory_warnings"]:
            print(f"• {w}")
        return

    # Default query execution
    query = args.query or "धान में पीला तना छेदक के लक्षण और रोकथाम बताएं?"
    agent = KrishiFarmerAgent()
    print("\n" + "=" * 60)
    print(f"🌾 KrishiMitra AI: Processing Query [{args.lang.upper()}]")
    print(f"Query: {query}")
    print("=" * 60)

    answer, meta = agent.ask(query, lang=args.lang)
    print("\n" + answer + "\n")

    chunks = meta.get("retrieved_chunks", [])
    if chunks:
        print("📚 Verified Provenance:")
        for idx, c in enumerate(chunks, 1):
            print(f" [{idx}] {c.get('source')} | {c.get('section')} (RRF: {c.get('rrf_score', 0):.4f})")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
