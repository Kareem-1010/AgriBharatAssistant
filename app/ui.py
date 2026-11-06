import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
from config import SUPPORTED_LANGUAGES, EVAL_RESULTS_PATH, EVAL_REPORT_PATH
from src.agents.farmer_agent import KrishiFarmerAgent
from src.agents.tools.npk_calculator import NPKFertilizerCalculator
from src.agents.tools.weather_tool import AgroWeatherAdvisor
from src.agents.tools.scheme_matcher import GovernmentSchemeMatcher

# Initialize the global agent
print("[UI] Initializing KrishiFarmerAgent for Gradio interface...")
agent = KrishiFarmerAgent()
npk_tool = NPKFertilizerCalculator()
weather_tool = AgroWeatherAdvisor()
scheme_matcher = GovernmentSchemeMatcher()

def process_farmer_query(message, audio_input, language_choice, voice_output_enabled):
    # Check if voice input was provided
    user_text = message
    if audio_input is not None and not message.strip():
        # Transcribe audio
        user_text = agent.asr.transcribe(audio_input, lang=language_choice)

    if not user_text.strip():
        return "कृपया अपना प्रश्न लिखें या बोलें (Please ask a question by typing or speaking).", None, ""

    # Execute agent pipeline
    response_text, meta = agent.ask(user_text, lang=language_choice, enable_tts=voice_output_enabled)

    # Citations markdown
    retrieved_chunks = meta.get("retrieved_chunks", [])
    if retrieved_chunks:
        citations = "### 📚 ICAR Source Documents Cited:\n"
        for idx, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "ICAR Guideline")
            section = chunk.get("section", "Crop Advisory")
            score = chunk.get("rrf_score", 0.0)
            citations += f"- **[{idx}]** `{source}` | *{section}* (RRF Relevance: {score:.4f})\n"
    else:
        citations = "ℹ️ Query answered by Guardrail / Direct Tool."

    audio_path = meta.get("audio_path")
    return response_text, audio_path, citations

def calculate_npk_ui(crop_name, acres, soil_status):
    res = npk_tool.calculate(crop_name, acres=float(acres), soil_fertility=soil_status)
    nutrients = res["target_nutrients_kg"]
    bags = res["commercial_bags_50kg"]
    
    summary = f"""### 🌾 {res['crop_name']} Fertilizer Schedule ({acres} Acres - {soil_status.capitalize()} Soil Fertility)

#### 🧪 Target Elemental Nutrients Required:
- **Nitrogen (N):** {nutrients['Nitrogen (N)']} kg
- **Phosphorus (P₂O₅):** {nutrients['Phosphorus (P2O5)']} kg
- **Potash (K₂O):** {nutrients['Potash (K2O)']} kg

---
#### 📦 Recommended Commercial 50kg Bags:
| Fertilizer | Total Quantity (kg) | Equivalent 50kg Bags |
|---|---|---|
| **Urea (46% N)** | {res['fertilizer_recommendation_kg']['Urea (46% N)']} kg | **{bags['Urea Bags']} Bags** |
| **DAP (18:46:0)** | {res['fertilizer_recommendation_kg']['DAP (18:46:0)']} kg | **{bags['DAP Bags']} Bags** |
| **MOP (60% K₂O)** | {res['fertilizer_recommendation_kg']['MOP (60% K2O)']} kg | **{bags['MOP Bags']} Bags** |

---
#### ⏰ Split Application Timing:
{res['application_schedule']}

> 💡 **Micronutrient Tip:** {res['micronutrient_tip']}
"""
    return summary

def check_weather_ui(temp, humidity, wind, rain):
    res = weather_tool.evaluate_spray_conditions(float(temp), float(humidity), float(wind), float(rain))
    verdict = res["spray_verdict"]
    color = "🟢" if verdict == "FAVORABLE" else "🔴"
    
    warnings = "\n".join([f"- ⚠️ {w}" for w in res["advisory_warnings"]])
    tips = "\n".join([f"- 💡 {t}" for t in res["operational_tips"]])
    
    md = f"""### {color} Agro-Weather Spray Recommendation: **{verdict}**

#### Current Parameters:
- **Temperature:** {temp} °C
- **Relative Humidity:** {humidity} %
- **Wind Speed:** {wind} km/h
- **Rain Probability:** {rain} %

#### Advisory Diagnostics:
{warnings}

#### Operational Spraying Best Practices:
{tips}
"""
    return md

def check_scheme_ui(is_landowner, is_govt_emp, pays_tax):
    res = scheme_matcher.check_eligibility(is_landowner, is_govt_emp, pays_tax)
    pmk_status = "✅ ELIGIBLE" if res["pm_kisan_eligible"] else "❌ INELIGIBLE"
    
    md = f"""### 🏛️ Government Farmer Scheme Eligibility Check

#### 1. PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)
- **Status:** **{pmk_status}**
- **Evaluation Notes:** {res['pm_kisan_notes']}
- **Annual Benefit:** ₹6,000 per year (in 3 installments of ₹2,000 via DBT)
- **Mandatory Requirements:** Aadhaar-bank seeding, e-KYC, and land record verification.

#### 2. PMFBY (Pradhan Mantri Fasal Bima Yojana)
- **Status:** ✅ Available for all farmers (loanee and non-loanee)
- **Farmer Premium:** 2.0% for Kharif, 1.5% for Rabi, 5.0% for Commercial/Horticulture.
- 🚨 **CRITICAL 72-HOUR RULE:** Report crop loss within 72 hours via Toll-Free 1800-180-1551 or Crop Insurance App.

#### 3. Kisan Credit Card (KCC)
- **Status:** ✅ Eligible
- **Interest Rate:** Effective 4.0% per annum (with prompt repayment incentive). Collateral-free up to ₹1.6 Lakh.
"""
    return md

def load_eval_summary():
    if not EVAL_RESULTS_PATH.exists():
        return "Benchmark has not been run yet. Run `python evaluation/run_evaluation.py` to generate metrics."

    with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("benchmark_metadata", {})
    agg = data.get("aggregate_metrics", {})
    cats = data.get("category_breakdown", {})
    langs = data.get("language_breakdown", {})

    md = f"""### 📊 Quantitative Evaluation Benchmark Results (45 Test Cases)
- **Evaluation Standard:** CS3235 Undergraduate Course Rubric
- **Mean Groundedness / Faithfulness:** **{agg.get('mean_groundedness', 0)*100:.1f}%**
- **Mean Retrieval Context Recall@K:** **{agg.get('mean_context_recall', 0)*100:.1f}%**
- **Mean Overall Composite Score:** **{agg.get('mean_overall_score', 0)*100:.1f}%**
- **Average Response Latency:** **{meta.get('average_latency_ms', 0):.1f} ms**

#### Performance by Agricultural Category:
| Category | Benchmark Score |
|---|---|
"""
    for c, s in cats.items():
        md += f"| `{c}` | **{s*100:.1f}%** |\n"

    md += """
#### Performance by Language:
| Language | Code | Benchmark Score |
|---|---|---|
"""
    for l, s in langs.items():
        name = {"hi": "Hindi", "en": "English", "te": "Telugu", "ta": "Tamil", "mr": "Marathi", "bn": "Bengali"}.get(l, l)
        md += f"| **{name}** | `{l}` | **{s*100:.1f}%** |\n"

    return md

def create_ui():
    custom_theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="amber",
        neutral_hue="slate"
    )

    with gr.Blocks(title="KrishiMitra AI | AI Farmer Advisory Assistant") as demo:
        gr.Markdown(
            """
            # 🌾 KrishiMitra AI: Multilingual AI Farmer Advisory Assistant
            ### *Powered by BharatGen AgriParam & AI4Bharat Indic Technologies | CS3235 Undergraduate AI Project*
            **Providing grounded Indian-language guidance for crop protection, irrigation, scientific fertilizer dosages, and government schemes.**
            """
        )

        with gr.Tabs():
            # Tab 1: Chat Assistant
            with gr.TabItem("💬 Farmer Advisory Chatbot"):
                with gr.Row():
                    with gr.Column(scale=2):
                        lang_dropdown = gr.Dropdown(
                            choices=[(f"{v['native']} ({v['name']})", k) for k, v in SUPPORTED_LANGUAGES.items()],
                            value="hi",
                            label="🌐 Select Advisory Language (भाषा चुनें)"
                        )
                        msg_input = gr.Textbox(
                            lines=3,
                            placeholder="धान में रोग लगा है, क्या करें? / How to manage pink bollworm in cotton? / ఎరువుల మోతాదు ఎంత?",
                            label="✍️ Farmer Query (लिखें या प्रश्न पूछें)"
                        )
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="🎙️ Voice Input (बोलकर पूछें - AI4Bharat Indic Speech)"
                        )
                        tts_checkbox = gr.Checkbox(
                            value=True,
                            label="🔊 Enable Vernacular Audio Response (ध्वनि उत्तर सक्षम करें)"
                        )
                        ask_btn = gr.Button("🌾 Get ICAR Expert Advisory (सलाह प्राप्त करें)", variant="primary")

                        gr.Markdown("#### 💡 Quick Sample Queries:")
                        btn_q1 = gr.Button("🌾 धान में पीला तना छेदक के लक्षण और रोकथाम?")
                        btn_q2 = gr.Button("💧 गेहूं में पहली सिंचाई (CRI) कब करनी चाहिए?")
                        btn_q3 = gr.Button("🐛 Cotton Pink Bollworm ETL and IPM spray?")
                        btn_q4 = gr.Button("🏛️ PMFBY फसल बीमा में नुकसान रिपोर्ट करने की समय सीमा?")

                    with gr.Column(scale=3):
                        response_output = gr.Markdown(label="Advisory Response", value="*आपकी कृषि सलाह यहां दिखाई देगी (Advisory will appear here)*")
                        audio_output = gr.Audio(label="🔊 Voice Response Playback", type="filepath")
                        citations_output = gr.Markdown(label="ICAR Source Citations")

                # Wiring
                ask_btn.click(
                    fn=process_farmer_query,
                    inputs=[msg_input, audio_input, lang_dropdown, tts_checkbox],
                    outputs=[response_output, audio_output, citations_output]
                )
                btn_q1.click(lambda: ("धान में पीला तना छेदक के लक्षण और रोकथाम बताएं?", "hi"), outputs=[msg_input, lang_dropdown])
                btn_q2.click(lambda: ("गेहूं में पहली सिंचाई (CRI) कितने दिन पर करें?", "hi"), outputs=[msg_input, lang_dropdown])
                btn_q3.click(lambda: ("What is the economic threshold level for pink bollworm in cotton and how to control it?", "en"), outputs=[msg_input, lang_dropdown])
                btn_q4.click(lambda: ("What is the maximum time limit to report localized crop loss under PMFBY?", "en"), outputs=[msg_input, lang_dropdown])

            # Tab 2: NPK Calculator Tool
            with gr.TabItem("🧪 NPK Fertilizer Calculator"):
                gr.Markdown("### 🧮 ICAR Scientific Fertilizer Formulation & Bag Calculator")
                with gr.Row():
                    crop_sel = gr.Dropdown(
                        choices=["Rice", "Wheat", "Bt Cotton", "Sugarcane", "Hybrid Maize", "Mustard", "Gram / Chickpea", "Tomato"],
                        value="Wheat",
                        label="🌱 Select Crop"
                    )
                    acres_num = gr.Number(value=2.0, label="📐 Land Area (Acres)", minimum=0.1, maximum=500.0)
                    soil_stat = gr.Radio(choices=["low", "medium", "high"], value="medium", label="🧪 Soil Fertility (Soil Health Card)")
                    calc_btn = gr.Button("Calculate Fertilizer Bags", variant="primary")

                npk_output = gr.Markdown()
                calc_btn.click(fn=calculate_npk_ui, inputs=[crop_sel, acres_num, soil_stat], outputs=npk_output)

            # Tab 3: Weather Advisor Tool
            with gr.TabItem("🌦️ Weather & Spray Advisor"):
                gr.Markdown("### 🌧️ Agricultural Operational Spray Decision Support")
                with gr.Row():
                    temp_slider = gr.Slider(0.0, 50.0, value=26.0, step=0.5, label="🌡️ Temperature (°C)")
                    humid_slider = gr.Slider(10.0, 100.0, value=65.0, step=1.0, label="💧 Humidity (%)")
                    wind_slider = gr.Slider(0.0, 40.0, value=8.0, step=0.5, label="💨 Wind Speed (km/h)")
                    rain_slider = gr.Slider(0.0, 100.0, value=15.0, step=1.0, label="🌧️ Rain Probability (%)")
                
                weather_btn = gr.Button("Evaluate Spray Safety", variant="primary")
                weather_output = gr.Markdown()
                weather_btn.click(fn=check_weather_ui, inputs=[temp_slider, humid_slider, wind_slider, rain_slider], outputs=weather_output)

            # Tab 4: Govt Welfare Schemes
            with gr.TabItem("🏛️ Government Schemes"):
                gr.Markdown("### 📋 Farmer Welfare Scheme Eligibility & Guidelines")
                with gr.Row():
                    chk_land = gr.Checkbox(value=True, label="Is landholder with registered land in name?")
                    chk_govt = gr.Checkbox(value=False, label="Is serving/retired government employee (except Group D)?")
                    chk_tax = gr.Checkbox(value=False, label="Paid Income Tax in last assessment year?")
                    scheme_btn = gr.Button("Check Scheme Eligibility", variant="primary")
                
                scheme_output = gr.Markdown()
                scheme_btn.click(fn=check_scheme_ui, inputs=[chk_land, chk_govt, chk_tax], outputs=scheme_output)

            # Tab 5: Evaluation & Error Analysis
            with gr.TabItem("📈 Evaluation & Rubric Benchmark"):
                gr.Markdown("### 🎯 Quantitative Benchmark Dashboard (45 Representative Test Cases)")
                eval_dashboard = gr.Markdown(value=load_eval_summary())
                refresh_eval_btn = gr.Button("🔄 Refresh Benchmark Metrics")
                refresh_eval_btn.click(fn=load_eval_summary, outputs=eval_dashboard)

        gr.Markdown(
            """
            ---
            *Academic Project for CS3235 (Undergraduate LLM / Multimodal AI) | Based on AI4Bharat & BharatGen Frameworks*
            """
        )

    return demo, custom_theme

if __name__ == "__main__":
    app, theme = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=theme)
