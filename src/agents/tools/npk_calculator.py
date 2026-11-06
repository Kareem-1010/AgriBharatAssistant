import math
from typing import Dict, Any

class NPKFertilizerCalculator:
    """
    Scientific Fertilizer Dosage & Bag Calculator based on ICAR Package of Practices.
    Calculates exact commercial fertilizer quantities (Urea, DAP, MOP, SSP)
    and split applications for given acreage and soil fertility status.
    """
    # Standard Recommended Doses of Fertilizer (N : P2O5 : K2O in kg per Acre)
    CROP_RDF_PER_ACRE = {
        "rice": {"name": "Rice (Paddy)", "N": 48.0, "P": 24.0, "K": 16.0, "splits": "Basal (25% N, 100% P, 50% K); Tillering (50% N); Panicle Initiation (25% N, 50% K)"},
        "wheat": {"name": "Wheat", "N": 48.0, "P": 24.0, "K": 16.0, "splits": "Basal (33% N, 100% P, 100% K); CRI stage (33% N); Tillering stage (33% N)"},
        "cotton": {"name": "Bt Cotton", "N": 48.0, "P": 24.0, "K": 24.0, "splits": "Basal (20% N, 100% P, 50% K); Squaring at 45 DAS (40% N); Flowering at 75 DAS (40% N, 50% K)"},
        "sugarcane": {"name": "Sugarcane", "N": 100.0, "P": 45.0, "K": 45.0, "splits": "Basal (25% N, 100% P, 50% K); 45 DAS (25% N); 90 DAS (25% N); 120 DAS before earthing up (25% N, 50% K)"},
        "maize": {"name": "Hybrid Maize", "N": 60.0, "P": 24.0, "K": 20.0, "splits": "Basal (33% N, 100% P, 100% K); Knee high stage (33% N); Tasseling stage (33% N)"},
        "mustard": {"name": "Mustard", "N": 32.0, "P": 16.0, "K": 16.0, "splits": "Basal (50% N, 100% P, 100% K, +10kg Sulphur); 30 DAS flowering initiation (50% N)"},
        "gram": {"name": "Gram / Chickpea", "N": 8.0, "P": 16.0, "K": 8.0, "splits": "100% Basal at sowing (Pulse crop fixes atmospheric N via Rhizobium)"},
        "tomato": {"name": "Tomato", "N": 60.0, "P": 40.0, "K": 40.0, "splits": "Basal (30% N, 100% P, 50% K); 30 DAT (35% N); 60 DAT fruit set (35% N, 50% K)"}
    }

    def calculate(self, crop: str, acres: float = 1.0, soil_fertility: str = "medium") -> Dict[str, Any]:
        crop_key = crop.lower().strip()
        if crop_key not in self.CROP_RDF_PER_ACRE:
            # find closest match
            for k in self.CROP_RDF_PER_ACRE:
                if k in crop_key:
                    crop_key = k
                    break
            else:
                crop_key = "rice"

        rdf = self.CROP_RDF_PER_ACRE[crop_key]
        
        # Soil test fertility multiplier
        multiplier = 1.0
        if soil_fertility.lower() == "low":
            multiplier = 1.25  # +25% for low fertility
        elif soil_fertility.lower() == "high":
            multiplier = 0.75  # -25% for high fertility

        target_N = rdf["N"] * acres * multiplier
        target_P = rdf["P"] * acres * multiplier
        target_K = rdf["K"] * acres * multiplier

        # Chemical formulation conversions
        # 1. DAP (18% N, 46% P2O5) supplies P2O5
        dap_kg = target_P / 0.46
        n_from_dap = dap_kg * 0.18

        # 2. Urea (46% N) supplies remaining N
        remaining_N = max(0.0, target_N - n_from_dap)
        urea_kg = remaining_N / 0.46

        # 3. MOP (60% K2O) supplies K
        mop_kg = target_K / 0.60

        # Commercial bag counts (50 kg bags)
        urea_bags = round(urea_kg / 50.0, 2)
        dap_bags = round(dap_kg / 50.0, 2)
        mop_bags = round(mop_kg / 50.0, 2)

        return {
            "crop_name": rdf["name"],
            "acres": acres,
            "soil_fertility": soil_fertility.capitalize(),
            "target_nutrients_kg": {
                "Nitrogen (N)": round(target_N, 1),
                "Phosphorus (P2O5)": round(target_P, 1),
                "Potash (K2O)": round(target_K, 1)
            },
            "fertilizer_recommendation_kg": {
                "Urea (46% N)": round(urea_kg, 1),
                "DAP (18:46:0)": round(dap_kg, 1),
                "MOP (60% K2O)": round(mop_kg, 1)
            },
            "commercial_bags_50kg": {
                "Urea Bags": urea_bags,
                "DAP Bags": dap_bags,
                "MOP Bags": mop_bags
            },
            "application_schedule": rdf["splits"],
            "micronutrient_tip": "Apply 10 kg/acre Zinc Sulphate (21%) as basal if soil zinc is deficient. Do not mix zinc with DAP."
        }

