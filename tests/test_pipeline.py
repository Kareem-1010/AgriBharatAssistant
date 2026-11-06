import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.chunker import AgriculturalChunker
from src.rag.guardrails import AgriculturalGuardrails
from src.agents.tools.npk_calculator import NPKFertilizerCalculator
from src.agents.tools.weather_tool import AgroWeatherAdvisor
from src.agents.tools.scheme_matcher import GovernmentSchemeMatcher
from config import RAW_DOCS_DIR

def test_document_chunking():
    chunker = AgriculturalChunker()
    chunks = chunker.process_all_documents(RAW_DOCS_DIR)
    assert len(chunks) >= 30, f"Expected at least 30 chunks, got {len(chunks)}"
    assert all("id" in c and "text" in c and "source" in c for c in chunks)

def test_npk_calculator():
    calc = NPKFertilizerCalculator()
    res = calc.calculate("wheat", acres=1.0, soil_fertility="medium")
    assert res["commercial_bags_50kg"]["Urea Bags"] > 0
    assert res["commercial_bags_50kg"]["DAP Bags"] > 0
    assert res["commercial_bags_50kg"]["MOP Bags"] > 0

def test_weather_spray_evaluator():
    advisor = AgroWeatherAdvisor()
    # High wind should be UNFAVORABLE
    res_bad = advisor.evaluate_spray_conditions(temp_c=25.0, humidity_pct=60.0, wind_kmh=22.0, rain_prob_pct=10.0)
    assert res_bad["spray_verdict"] == "UNFAVORABLE"

    # Mild conditions should be FAVORABLE
    res_good = advisor.evaluate_spray_conditions(temp_c=25.0, humidity_pct=50.0, wind_kmh=6.0, rain_prob_pct=5.0)
    assert res_good["spray_verdict"] == "FAVORABLE"

def test_scheme_matcher():
    matcher = GovernmentSchemeMatcher()
    res = matcher.check_eligibility(is_landowner=True, is_govt_employee=False, pays_income_tax=False)
    assert res["pm_kisan_eligible"] is True
    assert res["pmfby_eligible"] is True

def test_safety_guardrails():
    guard = AgriculturalGuardrails()
    # Hazardous chemical query must be blocked
    is_safe, msg = guard.check_query_safety("Can I spray acid or phenyl to kill weeds in wheat?")
    assert is_safe is False
    assert "SAFETY HAZARD" in msg

    # Out of domain human medical query must be blocked
    is_safe_med, msg_med = guard.check_query_safety("What medicine should I take for fever and headache?")
    assert is_safe_med is False
    assert "medical" in msg_med.lower() or "चिकित्सक" in msg_med
