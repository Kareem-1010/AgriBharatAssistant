import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from config import RAW_DOCS_DIR, SUPPORTED_LANGUAGES, TOP_K_RETRIEVAL
from src.rag.chunker import AgriculturalChunker
from src.rag.embeddings import MultilingualEmbeddings
from src.rag.hybrid_retriever import HybridAgriculturalRetriever
from src.rag.guardrails import AgriculturalGuardrails
from src.llm.agri_param import BharatGenAgriParamEngine
from src.translation.indic_trans import IndicTransManager
from src.speech.asr import IndicSpeechRecognizer
from src.speech.tts import IndicSpeechSynthesizer
from src.agents.tools.npk_calculator import NPKFertilizerCalculator
from src.agents.tools.weather_tool import AgroWeatherAdvisor
from src.agents.tools.scheme_matcher import GovernmentSchemeMatcher

class KrishiFarmerAgent:
    """
    End-to-End Multilingual AI Farmer Advisory Assistant.
    Coordinates Speech I/O, AI4Bharat Translation, Hybrid RAG Retrieval,
    Safety Guardrails, Agentic Tools, and BharatGen AgriParam Synthesis.
    """
    def __init__(self, raw_docs_dir: Path = RAW_DOCS_DIR):
        print("[KrishiAgent] Initializing AI Farmer Advisory Assistant...")
        self.chunker = AgriculturalChunker()
        self.chunks = self.chunker.process_all_documents(raw_docs_dir)
        print(f"[KrishiAgent] Loaded and chunked {len(self.chunks)} agricultural knowledge blocks.")

        self.embeddings = MultilingualEmbeddings()
        self.retriever = HybridAgriculturalRetriever(self.chunks, self.embeddings)
        self.guardrails = AgriculturalGuardrails()
        self.llm = BharatGenAgriParamEngine()
        self.translator = IndicTransManager()
        self.asr = IndicSpeechRecognizer()
        self.tts = IndicSpeechSynthesizer()

        # Tools
        self.npk_tool = NPKFertilizerCalculator()
        self.weather_tool = AgroWeatherAdvisor()
        self.scheme_tool = GovernmentSchemeMatcher()

        print("[KrishiAgent] All subsystems initialized successfully.")

    def detect_language(self, query: str) -> str:
        """
        Lightweight Unicode script detector for Indian languages.
        """
        # Devanagari range: \u0900-\u097F (Hindi / Marathi)
        # Telugu: \u0C00-\u0C7F
        # Tamil: \u0B80-\u0BFF
        # Bengali: \u0980-\u09FF
        counts = {"hi": 0, "te": 0, "ta": 0, "bn": 0, "mr": 0, "en": 0}
        for ch in query:
            code = ord(ch)
            if 0x0900 <= code <= 0x097F:
                counts["hi"] += 1
            elif 0x0C00 <= code <= 0x0C7F:
                counts["te"] += 1
            elif 0x0B80 <= code <= 0x0BFF:
                counts["ta"] += 1
            elif 0x0980 <= code <= 0x09FF:
                counts["bn"] += 1
            elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
                counts["en"] += 1

        top_lang = max(counts.keys(), key=lambda k: counts[k])
        if counts[top_lang] == 0:
            return "en"
        return top_lang

    def handle_voice_query(self, audio_file_path: str, user_lang: Optional[str] = None) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """
        Processes voice query: Audio -> ASR -> Query Processor -> TTS -> Response Audio
        """
        lang = user_lang or "hi"
        transcribed_text = self.asr.transcribe(audio_file_path, lang=lang)
        if not transcribed_text:
            return "Could not recognize audio input.", None, {}
        
        response_text, meta = self.ask(transcribed_text, lang=lang)
        audio_out = self.tts.synthesize(response_text, lang=lang)
        return response_text, audio_out, meta

    def ask(self, query: str, lang: Optional[str] = None, enable_tts: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Main query entrypoint for text queries.
        """
        detected_lang = lang or self.detect_language(query)

        # Step 1: Safety & Guardrail Check
        is_safe, safety_msg = self.guardrails.check_query_safety(query, lang=detected_lang)
        if not is_safe:
            return safety_msg, {
                "guardrail_status": "BLOCKED",
                "retrieved_chunks": [],
                "detected_lang": detected_lang,
                "audio_path": None
            }

        # Step 2: Agent Tool Intent Recognition (e.g., explicit NPK calculator or weather query)
        query_lower = query.lower()
        if ("calculate npk" in query_lower or "खाद की गणना" in query_lower or "fertilizer calculation" in query_lower) and any(c in query_lower for c in ["acre", "एकड़", "hectare"]):
            # Extract acres
            acres = 1.0
            num_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:acre|एकड़)', query_lower)
            if num_match:
                acres = float(num_match.group(1))
            calc_result = self.npk_tool.calculate("wheat" if "गेहूं" in query_lower or "wheat" in query_lower else "rice", acres=acres)
            msg = (
                f"### 🧪 NPK उर्वरक कैलकुलेटर परिणाम ({calc_result['crop_name']} - {acres} एकड़)\n\n"
                f"- **यूरिया (Urea - 46% N):** {calc_result['fertilizer_recommendation_kg']['Urea (46% N)']} kg ({calc_result['commercial_bags_50kg']['Urea Bags']} बैग)\n"
                f"- **डीएपी (DAP - 18:46:0):** {calc_result['fertilizer_recommendation_kg']['DAP (18:46:0)']} kg ({calc_result['commercial_bags_50kg']['DAP Bags']} बैग)\n"
                f"- **एमओपी (MOP - 60% K2O):** {calc_result['fertilizer_recommendation_kg']['MOP (60% K2O)']} kg ({calc_result['commercial_bags_50kg']['MOP Bags']} बैग)\n\n"
                f"**छिड़काव व अनुप्रयोग समय:** {calc_result['application_schedule']}\n\n"
                f"💡 {calc_result['micronutrient_tip']}"
            )
            return msg, {"guardrail_status": "PASSED", "tool_used": "npk_calculator", "retrieved_chunks": []}

        # Step 3: Hybrid Retrieval
        retrieved_chunks = self.retriever.retrieve(query, top_k=TOP_K_RETRIEVAL)

        # Step 4: Domain LLM Generation (BharatGen AgriParam)
        raw_response = self.llm.generate(query, retrieved_chunks, lang=detected_lang)

        # If safety warning was flagged during dosage check, prepend it
        if safety_msg:
            raw_response = f"{safety_msg}\n\n{raw_response}"

        # Step 5: Attach Verified Citations & Guardrail Disclaimers
        final_response = self.guardrails.attach_disclaimer_and_citations(raw_response, retrieved_chunks, lang=detected_lang)

        audio_path = None
        if enable_tts:
            audio_path = self.tts.synthesize(final_response, lang=detected_lang)

        metadata = {
            "guardrail_status": "PASSED" if not safety_msg else "FLAGGED_WARNING",
            "detected_lang": detected_lang,
            "retrieved_chunks": retrieved_chunks,
            "audio_path": audio_path,
            "top_citation": retrieved_chunks[0]["source"] if retrieved_chunks else None
        }

        return final_response, metadata
