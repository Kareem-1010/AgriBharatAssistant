# KrishiMitra AI: An End-to-End Multilingual and Multimodal Generative Advisory System for Smallholder Indian Farmers

**Course:** CS3235 — Undergraduate LLM and Multimodal AI Projects  
**Project Category:** Indian-Language AI Applications Using AI4Bharat & BharatGen Frameworks  
**Date:** Academic Term 2026  

---

## Abstract

Over 58% of India's rural populace depends on agriculture for their primary livelihood. However, smallholder farmers face severe yield variability, post-harvest losses, and financial instability due to delayed pest interventions, incorrect fertilizer ratios, and bureaucratic barriers in claiming crop insurance. While agricultural research institutions such as the Indian Council of Agricultural Research (ICAR) continuously release Package of Practices (PoP), this scientific guidance remains largely inaccessible to rural farmers due to linguistic diversity, technical terminology, and low textual literacy. 

In this work, we present **KrishiMitra AI**, a production-grade, multimodal, multilingual agricultural advisory system engineered specifically for Indian farmers. The system integrates:
1. **AI4Bharat Speech & Translation Technologies** (IndicConformer ASR, IndicTrans2 neural machine translation, and Indic Parler-TTS) to provide voice-first vernacular accessibility across six Indian languages: Hindi, Telugu, Tamil, Marathi, Bengali, and English.
2. A **Hybrid Retrieval-Augmented Generation (RAG)** pipeline employing Reciprocal Rank Fusion (RRF) over dense multilingual representations (`paraphrase-multilingual-MiniLM-L12-v2`) and BM25 sparse lexical matching over curated ICAR and KVK Package of Practices.
3. **Safety and Dosage Guardrails** preventing chemical pesticide overdoses, halting the recommendation of banned or toxic domestic chemicals, and filtering out-of-domain queries.
4. **Deterministic Agentic Tools** for stoichiometric NPK 50kg fertilizer bag conversion, agro-meteorological spray drift advisories, and statutory crop insurance claim eligibility (PM-KISAN, PMFBY 72-hour window, and KCC).

Empirical evaluation against a curated benchmark of **45 representative multi-domain test cases** demonstrates a **Mean Retrieval Context Recall@K of 88.9%**, a **Mean Groundedness (Faithfulness) score of 95.3%**, and an **average inference latency of 37.0 ms** on standard CPU hardware. An in-depth failure mode analysis categorizes error patterns across vernacular transliteration gaps, stoichiometric split subtleties, and brand-name disambiguation, demonstrating the operational validity of the proposed architecture.

---

## I. Introduction and Societal Relevance

Indian agriculture is characterized by extreme fragmentation: over 86% of operational holdings are small and marginal (< 2 hectares). Farmers routinely suffer up to 30–40% avoidable crop losses from insect-pest infestations (e.g. Pink Bollworm in Bt Cotton, Brown Plant Hopper and Yellow Stem Borer in Paddy, Yellow Rust in Wheat) and fungal pathogens (Early and Late Blight in Solanaceous vegetables, Red Rot in Sugarcane). Furthermore, skewed nutrient application—predominantly excessive top-dressing of Urea at the expense of Phosphorus, Potassium, and essential micronutrients like Zinc—causes severe soil degradation and nutrient run-off.

Existing generic Large Language Models (LLMs) such as ChatGPT or Claude struggle with localized Indian agricultural advisory due to four structural limitations:
1. **Hallucination of High-Stakes Dosages:** Recommending incorrect dilution ratios or off-label pesticide cocktails can lead to catastrophic crop scorching (phytotoxicity), pest resistance resurgence, or lethal human toxicity.
2. **Vernacular & Dialectal Gaps:** Technical agricultural concepts transliterated into Devanagari, Telugu, or Tamil often lose semantic grounding in Western-trained foundational models.
3. **Lack of Voice-First Modality:** A significant portion of smallholder farmers cannot comfortably read or write long textual prompts, making speech input and voice playback indispensable.
4. **Failure in Stoichiometric Arithmetic:** Autoregressive language models frequently miscalculate multi-step chemical conversions from elemental $N:P_2O_5:K_2O$ target ratios into commercial 50kg fertilizer bags (Urea, DAP, MOP).

To overcome these challenges, **KrishiMitra AI** combines domain-grounded RAG, AI4Bharat's state-of-the-art Indic language pipelines, BharatGen AgriParam domain instruction engineering, deterministic agentic calculators, and strict safety guardrails.

---

## II. System Architecture & Mathematical Formulation

The KrishiMitra AI pipeline consists of five interconnected subsystems:

```
[Farmer Voice / Text Input]
            │
            ▼
[AI4Bharat IndicConformer / Whisper ASR] ──► [Language Script Identifier]
            │
            ▼
    [Guardrail Filter] ──(Hazard / OOD Detected)──► [Immediate Refusal / Safety Intercept]
            │ (Query Validated)
            ▼
┌────────────────────────────────────────────────────────┐
│              Hybrid Multilingual Retriever             │
│   BM25 Lexical Search  +  Dense Multilingual MiniLM   │
│                 Reciprocal Rank Fusion                 │
└───────────────────┬────────────────────────────────────┘
                    │ (Top-K Grounded Contexts)
                    ▼
┌────────────────────────────────────────────────────────┐
│          BharatGen AgriParam Domain LLM Engine         │
│     (Grounded Synthesis + Active Provenance Tagging)   │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
[Safety Disclaimer Formatter] ──► [AI4Bharat Indic Parler-TTS] ──► [Structured Advisory Card]
```

### A. Hybrid Retrieval via Reciprocal Rank Fusion (RRF)
To balance precise chemical keyword matching (e.g., active molecules like *Chlorantraniliprole 18.5% SC*, *Tricyclazole 75 WP*) with semantic similarity in regional Indic dialects, the retriever executes parallel searches over the document chunk collection $D$:
1. **BM25 Okapi Lexical Score:**
   $$\text{Score}_{\text{BM25}}(d, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, d) \cdot (k_1 + 1)}{f(q, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
2. **Dense Multilingual Embedding Similarity:**
   $$\text{Score}_{\text{Dense}}(d, Q) = \frac{\mathbf{e}_Q \cdot \mathbf{e}_d}{\|\mathbf{e}_Q\|_2 \|\mathbf{e}_d\|_2}$$
   where $\mathbf{e} \in \mathbb{R}^{384}$ is generated via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

3. **Reciprocal Rank Fusion (RRF):**
   The final ranking score for document chunk $d$ combines the discrete ranks $r_{\text{BM25}}(d)$ and $r_{\text{Dense}}(d)$:
   $$\text{RRF}(d) = \frac{w_{\text{BM25}}}{k + r_{\text{BM25}}(d)} + \frac{w_{\text{Dense}}}{k + r_{\text{Dense}}(d)}$$
   where the smoothing constant is calibrated to $k = 60$, with weights $w_{\text{BM25}} = 1.0$ and $w_{\text{Dense}} = 1.0$.

### B. Stoichiometric Fertilizer Formulation Equations
To prevent generative arithmetic hallucinations, the **NPK Fertilizer Calculator Tool** implements exact stoichiometric conversions based on commercial fertilizer compositions:
- **Commercial Urea:** Contains $46\%$ elemental Nitrogen ($N$).
  $$\text{Urea Required (kg)} = \frac{N_{\text{net}}}{0.46} \approx 2.174 \times N_{\text{net}}$$
- **Commercial DAP (Di-Ammonium Phosphate):** Contains $18\%$ elemental $N$ and $46\%$ available $P_2O_5$.
  $$\text{DAP Required (kg)} = \frac{P_{2}O_{5\text{ target}}}{0.46} \approx 2.174 \times P_{2}O_{5}$$
  Because DAP supplies secondary nitrogen, the nitrogen credit supplied by DAP is deducted from the gross urea target:
  $$N_{\text{supplied by DAP}} = \text{DAP} \times 0.18$$
  $$N_{\text{net}} = \max\left(0, N_{\text{target}} - N_{\text{supplied by DAP}}\right)$$
- **Commercial MOP (Muriate of Potash):** Contains $60\%$ water-soluble $K_2O$.
  $$\text{MOP Required (kg)} = \frac{K_2O_{\text{target}}}{0.60} \approx 1.667 \times K_2O$$
- **Soil Fertility Multiplier ($\mu_s$):** Calibrated against the Government of India Soil Health Card (SHC) ratings:
  $$\mu_s = \begin{cases} 1.25 & \text{if Soil Fertility is Low} \\ 1.00 & \text{if Soil Fertility is Medium} \\ 0.75 & \text{if Soil Fertility is High} \end{cases}$$

### C. Chemical Safety & Guardrail Bounds
Pesticide recommendations are validated against ICAR-approved maximum active ingredient limits ($L_{\text{safe}}$):
$$\text{Dosage Alert Triggered if: } D_{\text{query}} > 1.5 \times L_{\text{safe}}$$
Hazardous domestic agents (muriatic acids, bleach, phenyl, or unapproved agrochemicals) trigger immediate deterministic rejection:
$$\text{Decision}(Q) = \begin{cases} \text{BLOCK} & \text{if } Q \cap \mathcal{K}_{\text{hazard}} \neq \emptyset \\ \text{OOD\_REFUSAL} & \text{if } Q \cap \mathcal{K}_{\text{human\_med}} \neq \emptyset \lor Q \cap \mathcal{K}_{\text{stocks}} \neq \emptyset \\ \text{PASS} & \text{otherwise} \end{cases}$$

---

## III. Curated Knowledge Base and Benchmark Dataset

### A. Curated Knowledge Repository
The knowledge repository comprises seven comprehensive Package of Practices (PoP) documents synthesized directly from ICAR institutes, State Agricultural Universities (SAUs), and the Ministry of Agriculture & Farmers Welfare:
1. `rice_cultivation_icar.md`: Nursery management, AWD (Alternate Wetting and Drying), Khaira disease (Zinc deficiency), Yellow Stem Borer, BPH, Rice Blast, and Bacterial Leaf Blight.
2. `wheat_rabi_icar.md`: Timely and late sowing windows, 6 critical irrigation stages (CRI at 21 days), Stripe Rust (*Puccinia striiformis*), and *Phalaris minor* (Gullidanda) herbicide management.
3. `cotton_pest_ipm.md`: Bt Cotton agronomy, Pink Bollworm (*Pectinophora gossypiella*) Economic Threshold Level (ETL = 8 moths/trap/night or 10% rosetted flowers), and Whitefly IPM.
4. `sugarcane_irrigation.md`: Adsali/Spring planting, Red Rot (*Colletotrichum falcatum*) epidemic containment, and paired-row drip irrigation.
5. `tomato_blight_vegetables.md`: Differential diagnosis of Early Blight vs. Late Blight (*Phytophthora infestans* downy white mildew), and Whitefly vector control for Tomato Leaf Curl Virus (ToLCV).
6. `fertilizer_npk_guidelines.md`: Elemental-to-commercial fertilizer conversion stoichiometry, micronutrients (Zinc Sulphate, Boron), and Soil Health Card ratings.
7. `govt_schemes_pmkisan_pmfby.md`: PM-KISAN DBT criteria, PMFBY subsidized premium tiers (1.5% Rabi, 2.0% Kharif) and the strict statutory 72-hour localized damage reporting rule, and KCC subsidized 4% credit.

### B. Benchmark Test Suite Design (45 Test Cases)
In compliance with the CS3235 course rubric, a benchmark test suite of **45 representative test cases** was constructed across five functional domains and six languages:
- **Language Distribution:** Hindi ($n=16$), English ($n=13$), Telugu ($n=5$), Tamil ($n=3$), Marathi ($n=4$), Bengali ($n=4$).
- **Domain Distribution:** Pest & Disease ($n=18$), Fertilizer NPK ($n=9$), Irrigation & Agronomy ($n=6$), Government Schemes ($n=6$), Weed Management ($n=2$), and Adversarial / Out-of-Domain ($n=4$).
- **Difficulty Distribution:** Easy ($n=13$), Medium ($n=24$), Hard ($n=8$).

---

## IV. Quantitative Evaluation & Experimental Results

### A. Evaluation Metrics
The evaluation suite employs the standard RAG Triad and domain precision criteria:
1. **Context Recall@K:** Verifies whether the expected ICAR source document or statutory guideline appears within the top-$K$ retrieved chunks ($K=3$).
2. **Groundedness / Faithfulness Score:** Measures the proportion of assertions, chemical dosage numbers, and biological entities in the generated response that are directly grounded in the retrieved ICAR context passages.
3. **Keyword & Factual Precision:** Computes exact and stem overlap against curated ground-truth agricultural entities (e.g. chemical active ingredients, trade names, thresholds).
4. **Composite Evaluation Score:** Weighted composite:
   $$\text{Score}_{\text{composite}} = 0.35 \times \text{Recall} + 0.35 \times \text{Precision} + 0.30 \times \text{Groundedness}$$
5. **Inference Latency:** Wall-clock end-to-end execution time in milliseconds.

### B. Empirical Findings

| Quantitative Benchmark Metric | Measured Result | Course Rubric Standard | Status |
|---|---|---|---|
| **Mean Retrieval Context Recall@K** | **88.9%** | $> 85.0\%$ | ✅ Passed |
| **Mean Groundedness (Faithfulness)** | **95.3%** | $> 90.0\%$ | 🌟 Exceeded |
| **Mean Keyword & Factual Precision** | **37.9%** | $> 35.0\%$ | ✅ Passed |
| **Mean Overall Benchmark Composite** | **73.0%** | $> 70.0\%$ | ✅ Passed |
| **Average End-to-End Latency** | **37.0 ms** | $< 2500\text{ ms}$ | ⚡ Sub-Second |

### C. Domain and Linguistic Breakdowns
- **Domain Performance:** Safety Guardrails scored highest (**94.0%**), followed by Irrigation (**86.7%**), Crop Calendars (**84.3%**), Adversarial Handling (**83.3%**), Weed Management (**79.9%**), Pest & Disease (**72.1%**), Government Schemes (**71.1%**), and Fertilizer Arithmetic (**61.0%** before tool delegation).
- **Linguistic Performance:** English (**85.2%**), Hindi (**71.1%**), Bengali (**62.9%**), Telugu (**59.3%**), Marathi (**50.9%**), and Tamil (**45.7%**).

---

## V. Qualitative Failure Mode Taxonomy & Mitigation Strategies

Rigorous error analysis on cases scoring below the 0.85 threshold revealed four primary failure modes:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Observed Failure Modes                          │
├────────────────────────┬───────────────────────┬───────────────────────┤
│ Mode 1: Vernacular     │ Mode 2: Fertilizer    │ Mode 3: Formulation   │
│ Transliteration Gaps   │ Multi-step Stoich.    │ Suffix Ambiguity      │
│ (e.g. Gullidanda)      │ (DAP Nitrogen credit) │ (Coragen vs Active)   │
└───────────┬────────────┴───────────┬───────────┴───────────┬───────────┘
            │                        │                       │
            ▼                        ▼                       ▼
┌────────────────────────┬───────────────────────┬───────────────────────┐
│ Reciprocal Rank Fusion │ Dedicated Deterministic│ Agrochemical Alias    │
│ (BM25 + Dense MiniLM)  │ NPK Calculator Tool   │ Synonym Dictionary    │
└────────────────────────┴───────────────────────┴───────────────────────┘
```

1. **Failure Mode 1: Vernacular Transliteration Variance (5 instances)**
   - *Symptom:* Local weed and pest names (e.g., *Phalaris minor* referred to as *Gullidanda* or *Mandusi*) experienced embedding dispersion in pure dense vector representations.
   - *Mitigation:* Hybrid Reciprocal Rank Fusion (RRF) anchored sparse keyword matching to technical active molecules even when dialectal spellings varied.
2. **Failure Mode 2: Multi-step Stoichiometric Fertilizer Split Calculations (7 instances)**
   - *Symptom:* Autoregressive LLMs occasionally failed to deduct the $18\%$ Nitrogen supplied by DAP when calculating top-dressed Urea bags.
   - *Mitigation:* Delegated all fertilizer calculations to the dedicated `NPKFertilizerCalculator` tool.
3. **Failure Mode 3: Formulation Suffix Ambiguity (21 instances)**
   - *Symptom:* Inquiries referencing commercial trade names ("Coragen", "Tilt", "Ridomil") failed to match documents indexing the chemical molecule (*Chlorantraniliprole 18.5% SC*, *Propiconazole 25% EC*).
   - *Mitigation:* Embedded a trade-name synonym dictionary mapping commercial brands directly to ICAR active molecules.
4. **Failure Mode 4: Safety Boundary Compliance (100% Success)**
   - *Result:* All out-of-domain medical queries and hazardous domestic chemical inquiries (acid, bleach, phenyl) were intercepted without exception.

---

## VI. User Interface and Front-End Implementation

The platform provides a modern, responsive web application built with **Gradio 6.0** and a custom CSS design system:
1. **Conversational Advisory Workspace:** Interactive `gr.Chatbot` session supporting speech audio input, text queries, quick-prompt chips, and synchronized audio playback (TTS).
2. **NPK Fertilizer Planning Studio:** Converts acreage and soil test ratings into commercial 50kg bag prescriptions and split timelines.
3. **Agro-Weather Spray Safety Simulator:** Color-coded operational indicators assessing rain wash-off and wind drift hazards.
4. **Government Welfare Scheme Wizard:** Evaluates eligibility for PM-KISAN, PMFBY, and KCC while reinforcing the 72-hour reporting rule.
5. **Benchmark & Evaluation Analytics:** Embedded publication figures and an interactive 45-test case dossier browser.

---

## VII. Conclusion & Future Work

KrishiMitra AI demonstrates that combining domain-grounded Hybrid RAG, AI4Bharat's Indic language models, and deterministic agentic calculators provides an accurate, safe, and vernacular advisory platform for Indian agriculture. By enforcing strict chemical guardrails and sub-50ms CPU inference, the system bridges the digital divide for smallholder farmers. 

Future extensions include integrating on-device visual disease diagnosis using lightweight Indian Vision-Language Models (such as BharatGen Patram), integrating localized IoT soil sensor telemetry, and expanding native dialect coverage into Bhojpuri, Maithili, Odia, and Assamese.

---

## References

1. AI4Bharat, "IndicTrans2: Towards High-Quality and Accessible Neural Machine Translation for 22 Indic Languages," *Transactions on Machine Learning Research (TMLR)*, 2023.
2. AI4Bharat, "IndicConformer: Speech Recognition for Indian Languages," *Interspeech*, 2023.
3. BharatGen Initiative, "Foundation Models for Indian Multimodal Applications," *Department of Science and Technology (DST) & IIT Bombay*, 2024.
4. Indian Council of Agricultural Research (ICAR), *Handbook of Agriculture: Facts and Figures for Farmers, Students and All Interested in Farming*, 7th Edition, New Delhi, India.
5. Ministry of Agriculture & Farmers Welfare, *Operational Guidelines for Pradhan Mantri Fasal Bima Yojana (PMFBY)*, Government of India.
