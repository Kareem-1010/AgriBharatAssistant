import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_SUITE_PATH = DATA_DIR / "test_suite_50_cases.json"
EVALUATION_DIR = BASE_DIR / "evaluation"
EVAL_RESULTS_PATH = EVALUATION_DIR / "evaluation_results.json"
EVAL_REPORT_PATH = EVALUATION_DIR / "failure_analysis_report.md"

# Supported Indian Languages (ISO 639-1 code -> Full name & Indic script name)
SUPPORTED_LANGUAGES = {
    "hi": {"name": "Hindi", "native": "हिन्दी", "indic_code": "hin_Deva"},
    "en": {"name": "English", "native": "English", "indic_code": "eng_Latn"},
    "te": {"name": "Telugu", "native": "తెలుగు", "indic_code": "tel_Telu"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "indic_code": "tam_Taml"},
    "mr": {"name": "Marathi", "native": "मराठी", "indic_code": "mar_Deva"},
    "bn": {"name": "Bengali", "native": "বাংলা", "indic_code": "ben_Beng"},
}

DEFAULT_LANGUAGE = "hi"

# RAG & Retrieval Configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_RETRIEVAL = 3
CHUNK_SIZE = 450
CHUNK_OVERLAP = 60
RRF_K = 60  # Reciprocal Rank Fusion constant

# LLM & Reasoning Engine
# Supports: "bharatgen-agriparam", "groq", "gemini", "ollama", "local-huggingface", "grounded-fallback"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "grounded-agriparam")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Chemical Safety & Guardrail Settings
MAX_ALLOWABLE_PESTICIDE_FACTOR = 1.5  # Reject dosages > 1.5x ICAR recommendation
DISCLAIMER_TEXT = {
    "en": "⚠️ ICAR Scientific Advisory: Follow recommended pesticide dosages. Wear protective gear when spraying.",
    "hi": "⚠️ आईसीएआर वैज्ञानिक सलाह: कीटनाशकों की संस्तुत मात्रा का ही उपयोग करें। छिड़काव के समय सुरक्षा उपकरण पहनें।",
    "te": "⚠️ ICAR శాస్త్రీయ సలహా: పురుగుమందుల సిఫార్సు మోతాదులను ఖచ్చితంగా పాటించండి.",
    "ta": "⚠️ ICAR அறிவியல் ஆலோசனை: பரிந்துரைக்கப்பட்ட பூச்சிக்கொல்லி அளவுகளை மட்டுமே பயன்படுத்தவும்.",
    "mr": "⚠️ ICAR शास्त्रीय सल्ला: शिफारस केलेल्या कीटकनाशक प्रमाणाचाच वापर करा.",
    "bn": "⚠️ ICAR বৈজ্ঞানিক পরামর্শ: শুধুমাত্র অনুমোদিত কীটনাশকের মাত্রা ব্যবহার করুন।"
}
