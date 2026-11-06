import os
import re
from typing import List, Dict, Any, Optional

class BharatGenAgriParamEngine:
    """
    BharatGen AgriParam Agricultural Domain LLM Engine.
    Employs domain-specific prompt templates, grounded factual synthesis,
    and vernacular alignment for Indian agriculture.
    Supports external LLM backends (Groq, Gemini, Ollama) and a standalone
    grounded synthesis engine for deterministic evaluation.
    """
    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self.system_prompt = (
            "You are BharatGen AgriParam, India's specialized Agricultural AI Assistant trained on "
            "ICAR (Indian Council of Agricultural Research), KVK package of practices, and Ministry of Agriculture guidelines.\n"
            "Rules:\n"
            "1. Ground all recommendations strictly in the verified context provided.\n"
            "2. Always specify exact dosages, scientific names, timing, and split applications.\n"
            "3. Include safety warnings for chemical pesticides.\n"
            "4. Respond respectfully and clearly in the user's chosen language.\n"
            "5. If a query is outside agriculture or dangerous, refuse politely."
        )

    def _call_groq(self, prompt: str, groq_api_key: str) -> Optional[str]:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[AgriParam] Groq call failed: {e}")
            return None

    def _call_gemini(self, prompt: str, gemini_api_key: str) -> Optional[str]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"{self.system_prompt}\n\nUser Query & Context:\n{prompt}")
            return response.text
        except Exception as e:
            print(f"[AgriParam] Gemini call failed: {e}")
            return None

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]], lang: str = "en") -> str:
        # Check API keys first
        groq_key = os.getenv("GROQ_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")

        context_str = "\n\n".join([f"Source: {c['source']} ({c['section']})\n{c['text']}" for c in retrieved_chunks])
        user_prompt = f"Context:\n{context_str}\n\nFarmer Query ({lang}): {query}\n\nPlease provide a clear, grounded agricultural advisory."

        if groq_key:
            res = self._call_groq(user_prompt, groq_key)
            if res:
                return res

        if gemini_key:
            res = self._call_gemini(user_prompt, gemini_key)
            if res:
                return res

        # Standalone Grounded AgriParam Synthesis Engine
        return self._grounded_synthesis(query, retrieved_chunks, lang)

    def _grounded_synthesis(self, query: str, retrieved_chunks: List[Dict[str, Any]], lang: str) -> str:
        """
        Grounded domain synthesis extracting facts, dosages, symptoms, and guidelines
        from retrieved ICAR chunks without external API dependencies.
        """
        if not retrieved_chunks:
            no_info_msgs = {
                "en": "I could not find verified ICAR advisory documentation for this specific query. Please consult your local Krishi Vigyan Kendra (KVK).",
                "hi": "इस विशिष्ट प्रश्न के लिए आईसीएआर के संस्तुत दस्तावेज नहीं मिले। कृपया अपने नजदीकी कृषि विज्ञान केंद्र (केवीके) से संपर्क करें।",
                "te": "ఈ ప్రశ్నకు సంబంధించిన ICAR ఆధారాలు లభించలేదు. దయచేసి స్థానిక కృషి విజ్ఞాన కేంద్రాన్ని సంప్రదించండి.",
                "ta": "இந்த கேள்விக்கான ICAR ஆவணங்கள் கிடைக்கவில்லை. உங்கள் பகுதி வேளாண் அறிவியல் மையத்தை அணுகவும்.",
                "mr": "या प्रश्नासाठी ICAR चे अधिकृत दस्तऐवज उपलब्ध नाहीत. कृपया स्थानिक कृषी विज्ञान केंद्राशी संपर्क साधा.",
                "bn": "এই প্রশ্নের জন্য ICAR নির্দেশিকা পাওয়া যায়নি। অনুগ্রহ করে স্থানীয় কৃষি বিজ্ঞান কেন্দ্রে যোগাযোগ করুন।"
            }
            return no_info_msgs.get(lang, no_info_msgs["en"])

        primary_chunk = retrieved_chunks[0]
        text = primary_chunk["text"]
        section = primary_chunk["section"]
        source = primary_chunk["source"]

        # Parse key agricultural entities and sentences
        lines = [line.strip() for line in text.split("\n") if line.strip() and not line.startswith("[")]
        
        # Select informative lines with actionable advice, numbers, chemicals
        informative_lines = []
        for l in lines:
            clean_l = re.sub(r'^[#\*\-\d\.]+\s*', '', l).strip()
            if len(clean_l) > 15:
                informative_lines.append(clean_l)

        points = [f"• {l}" for l in informative_lines[:5]]
        response_body = "\n".join(points) if points else text[:400]

        if lang == "hi":
            return (
                f"### 🌾 भारतजेन एग्री-परम (BharatGen AgriParam) कृषि परामर्श\n\n"
                f"**संबंधित विषय:** {section}\n\n"
                f"**आईसीएआर (ICAR) संस्तुत प्रमुख वैज्ञानिक निर्देश:**\n"
                f"{response_body}\n\n"
                f"**कार्यान्वयन सुझाव:**\n"
                f"1. खेत में रोग या कीट के शुरुआती लक्षण दिखते ही अनुशंसित मात्रा में छिड़काव करें।\n"
                f"2. कीटनाशक का घोल बनाते समय साफ पानी (लगभग 150-200 लीटर प्रति एकड़) का उपयोग करें।\n"
                f"3. तेज धूप या वर्षा की संभावना होने पर छिड़काव स्थगित करें।"
            )
        elif lang == "te":
            return (
                f"### 🌾 భారత్‌జెన్ అగ్రి-పారమ్ (BharatGen AgriParam) వ్యవసాయ సలహా\n\n"
                f"**విషయం:** {section}\n\n"
                f"**ICAR సిఫార్సు చేసిన ముఖ్యమైన శాస్త్రీయ మార్గదర్శకాలు:**\n"
                f"{response_body}\n\n"
                f"**ఆచరణాత్మక సూచనలు:**\n"
                f"1. తెగులు ప్రారంభ దశలోనే సిఫార్సు చేసిన మోతాదులో పిచికారీ చేయండి.\n"
                f"2. ఎకరానికి 150-200 లీటర్ల శుభ్రమైన నీటిని మాత్రమే ఉపయోగించండి."
            )
        elif lang == "ta":
            return (
                f"### 🌾 பாரத்ஜென் அக்ரி-பராம் (BharatGen AgriParam) வேளாண் ஆலோசனை\n\n"
                f"**தலைப்பு:** {section}\n\n"
                f"**ICAR பரிந்துரைத்த முக்கிய அறிவியல் வழிகாட்டுதல்கள்:**\n"
                f"{response_body}\n\n"
                f"**பயனுள்ள குறிப்புகள்:**\n"
                f"1. நோய் தொடக்க நிலையிலேயே பரிந்துரைக்கப்பட்ட மருந்தை தெளிக்கவும்.\n"
                f"2. ஒரு ஏக்கருக்கு 150-200 லிட்டர் சுத்தமான தண்ணீரை பயன்படுத்தவும்."
            )
        elif lang == "mr":
            return (
                f"### 🌾 भारतजेन ॲग्री-पॅराम (BharatGen AgriParam) कृषी सल्ला\n\n"
                f"**विषय:** {section}\n\n"
                f"**ICAR शिफारस केलेल्या महत्त्वाच्या शास्त्रीय सूचना:**\n"
                f"{response_body}\n\n"
                f"**महत्त्वाच्या टिप्स:**\n"
                f"1. रोगाची किंवा किडीची लक्षणे दिसताच वेळेवर योग्य प्रमाणात फवारणी करावी.\n"
                f"2. फवारणीसाठी प्रति एकर १५०-२०० लिटर स्वच्छ पाणी वापरावे."
            )
        elif lang == "bn":
            return (
                f"### 🌾 ভারতজেন এগ্রি-প্যারাম (BharatGen AgriParam) কৃষি পরামর্শ\n\n"
                f"**বিষয়:** {section}\n\n"
                f"**ICAR অনুমোদিত গুরুত্বপূর্ণ বৈজ্ঞানিক নির্দেশিকা:**\n"
                f"{response_body}\n\n"
                f"**পরামর্শ:**\n"
                f"1. রোগের লক্ষণ দেখা মাত্রই অনুমোদিত মাত্রায় ওষুধ স্প্রে করুন।\n"
                f"2. একর প্রতি ১৫০-২০০ লিটার পরিষ্কার জল ব্যবহার করুন।"
            )
        else: # English default
            return (
                f"### 🌾 BharatGen AgriParam Advisory\n\n"
                f"**Topic / Section:** {section}\n\n"
                f"**Verified ICAR Scientific Recommendations:**\n"
                f"{response_body}\n\n"
                f"**Actionable Field Instructions:**\n"
                f"1. Apply the recommended interventions at the exact Economic Threshold Level (ETL) or symptom onset.\n"
                f"2. Ensure proper calibration with 150-200 Litres of clean water per acre using appropriate flat-fan or cone nozzles.\n"
                f"3. Strictly avoid spraying during midday hours or windy conditions."
            )
