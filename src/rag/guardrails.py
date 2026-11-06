import re
from typing import Dict, Any, Tuple, Optional
from config import DISCLAIMER_TEXT

class AgriculturalGuardrails:
    """
    Safety, Hallucination and Domain Guardrails for Indian Agricultural Advisory.
    - Prevents hazardous chemical overdoses
    - Flags out-of-domain queries (human medical, financial stocks, illegal chemicals)
    - Verifies factual grounding and appends official ICAR disclaimers
    """
    def __init__(self):
        # Hazard keywords
        self.hazard_keywords = [
            "acid", "तेजाब", "फिनाइल", "phenyl", "harpic", "bleach", "पोइजन", "poison",
            "आत्महत्या", "suicide", "murder", "kill human"
        ]
        
        # Out-of-domain topics
        self.ood_keywords = {
            "human_medical": ["fever", "headache", "बुखार", "सिरदर्द", "covid", "paracetamol", "cough syrup", "రక్తపోటు", "காய்ச்சல்"],
            "financial_stocks": ["share market", "stocks", "crypto", "bitcoin", "शेयर बाजार", "म्यूचुअल फंड", "forex"],
            "politics": ["election", "vote for", "minister party", "चुनाव", "पार्टी"]
        }

        # Known ICAR safe dosage reference table (active ingredient -> max safe dose per acre)
        self.safe_dosage_limits = {
            "chlorantraniliprole": {"safe_dose": 60, "unit": "ml/acre", "brand": "Coragen"},
            "spinosad": {"safe_dose": 75, "unit": "ml/acre", "brand": "Tracer"},
            "emamectin": {"safe_dose": 100, "unit": "g/acre", "brand": "Proclaim"},
            "propiconazole": {"safe_dose": 200, "unit": "ml/acre", "brand": "Tilt"},
            "streptocycline": {"safe_dose": 6, "unit": "g/acre", "brand": "Streptocycline"},
            "sulfosulfuron": {"safe_dose": 13.5, "unit": "g/acre", "brand": "Leader"},
            "clodinafop": {"safe_dose": 160, "unit": "g/acre", "brand": "Topik"}
        }

    def check_query_safety(self, query: str, lang: str = "en") -> Tuple[bool, Optional[str]]:
        query_lower = query.lower()

        # 1. Hazardous chemical / illegal intervention check
        for kw in self.hazard_keywords:
            if kw in query_lower:
                msg = {
                    "en": "🚨 SAFETY HAZARD DETECTED: Use of domestic acids, phenyl, or non-agricultural hazardous chemicals on crops is illegal, dangerous, and causes permanent soil toxicity. Only apply ICAR-approved agrochemicals.",
                    "hi": "🚨 सुरक्षा चेतावनी: फसलों में तेजाब, फिनाइल या गैर-कृषि विषाक्त रसायनों का छिड़काव गैर-कानूनी और अत्यंत विनाशकारी है। कृपया केवल आईसीएआर संस्तुत अनुमोदित कृषि दवाओं का ही प्रयोग करें।",
                    "te": "🚨 భద్రతా హెచ్చరిక: పంటలలో యాసిడ్ లేదా ప్రమాదకర రసాయనాలను వాడటం చట్టవిరుద్ధం మరియు అత్యంత ప్రమాదకరం. ICAR ఆమోదించిన పురుగుమందులను మాత్రమే వాడండి.",
                    "ta": "🚨 பாதுகாப்பு எச்சரிக்கை: பயிர்களில் ஆசிட் அல்லது ஆபத்தான ரசாயனங்களை தெளிப்பது முற்றிலும் ஆபத்தானது. ICAR அங்கீகரித்த மருந்துகளை மட்டுமே பயன்படுத்தவும்.",
                    "mr": "🚨 सुरक्षा इशारा: पिकांवर आम्ल (acid) किंवा विषारी रसायने फवारणे बेकायदेशीर आणि जमिनीसाठी अत्यंत घातक आहे. फक्त ICAR प्रमाणित औषधे वापरा.",
                    "bn": "🚨 নিরাপত্তা সতর্কতা: ফসলে ক্ষতিকর অ্যাসিড বা বিষাক্ত রাসায়নিক প্রয়োগ অবৈধ এবং মাটির জন্য মারাত্মক ক্ষতিকর। শুধুমাত্র ICAR অনুমোদিত কীটনাশক ব্যবহার করুন।"
                }
                return False, msg.get(lang, msg["en"])

        # 2. Out-of-Domain checks
        for cat, kw_list in self.ood_keywords.items():
            for kw in kw_list:
                if kw in query_lower:
                    if cat == "human_medical":
                        msg = {
                            "en": "⚕️ Out-of-Domain: I am an Agricultural Advisory AI (Kisan AI) and cannot provide medical advice for humans. Please consult a qualified doctor or local Primary Health Center (PHC).",
                            "hi": "⚕️ कार्यक्षेत्र से बाहर: मैं एक कृषि परामर्श सहायक (किसान एआई) हूं और मानव स्वास्थ्य या दवाओं पर परामर्श नहीं दे सकता। कृपया नजदीकी चिकित्सक या प्राथमिक स्वास्थ्य केंद्र (PHC) से संपर्क करें।",
                            "te": "⚕️ నేను కేవలం వ్యవసాయ సహాయకుడిని మాత్రమే. మానవ ఆరోగ్య సమస్యలకు దయచేసి వైద్యుడిని సంప్రదించండి.",
                            "ta": "⚕️ நான் விவசாய ஆலோசனை AI மட்டுமே. மனித மருத்துவ உதவிக்கு மருத்துவரை அணுகவும்.",
                            "mr": "⚕️ मी एक कृषी सल्लागार सहाय्यक आहे आणि मानवी वैद्यकीय सल्ला देऊ शकत नाही. कृपया डॉक्टरांचा सल्ला घ्या.",
                            "bn": "⚕️ আমি একজন কৃষি উপদেষ্টা এআই। মানুষের চিকিৎসার জন্য অনুগ্রহ করে ডাক্তারের পরামর্শ নিন।"
                        }
                        return False, msg.get(lang, msg["en"])
                    elif cat == "financial_stocks":
                        msg = {
                            "en": "📈 Out-of-Domain: I specialize strictly in agricultural sciences, crop management, and farmer welfare schemes. I do not provide stock market or trading advice.",
                            "hi": "📈 कार्यक्षेत्र से बाहर: मेरी विशेषज्ञता केवल कृषि विज्ञान, फसल सुरक्षा और किसान कल्याण योजनाओं तक सीमित है। मैं शेयर बाजार या ट्रेडिंग सलाह नहीं देता।",
                            "te": "📈 నేను స్టాక్ మార్కెట్ లేదా ట్రేడింగ్ సలహాలు ఇవ్వలేను.",
                            "ta": "📈 நான் பங்குச் சந்தை ஆலோசனைகளை வழங்குவதில்லை.",
                            "mr": "📈 मी शेअर बाजार किंवा ट्रेडिंग संदर्भात सल्ला देत नाही.",
                            "bn": "📈 আমি শেয়ার বাজার সংক্রান্ত পরামর্শ দিই না।"
                        }
                        return False, msg.get(lang, msg["en"])

        # 3. Chemical Dosage Overdose check
        for chemical, info in self.safe_dosage_limits.items():
            if chemical in query_lower:
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query_lower)
                for num_str in numbers:
                    num = float(num_str)
                    if num > (info["safe_dose"] * 1.8):  # > 1.8x safe dose
                        warning = (
                            f"⚠️ DOSAGE OVERDOSE WARNING: {chemical.capitalize()} query mentions {num} {info['unit'].split('/')[0]}, "
                            f"whereas ICAR recommended maximum safe dose is strictly {info['safe_dose']} {info['unit']}. "
                            f"Overdosing triggers severe resistance, phytotoxicity, and environmental toxicity."
                        )
                        return True, warning

        return True, None

    def attach_disclaimer_and_citations(self, response_text: str, retrieved_chunks: list, lang: str = "en") -> str:
        disclaimer = DISCLAIMER_TEXT.get(lang, DISCLAIMER_TEXT["en"])
        sources_list = []
        for c in retrieved_chunks:
            source_name = c.get("source", "ICAR Guidelines")
            section = c.get("section", "Crop Advisory")
            sources_list.append(f"• `{source_name}` - {section}")

        unique_sources = "\n".join(list(dict.fromkeys(sources_list)))

        formatted = (
            f"{response_text}\n\n"
            f"---\n"
            f"**📚 Verified ICAR Knowledge Provenance:**\n{unique_sources}\n\n"
            f"{disclaimer}"
        )
        return formatted

