from typing import Dict, Any, List

class GovernmentSchemeMatcher:
    """
    Government Agricultural Welfare Scheme & Eligibility Checker.
    Provides eligibility validation, benefit breakdown, documentation checklists,
    and emergency helpline guidance for PM-KISAN, PMFBY, and KCC.
    """
    SCHEMES = {
        "PM-KISAN": {
            "name": "Pradhan Mantri Kisan Samman Nidhi",
            "benefit": "₹6,000 per year in 3 equal installments of ₹2,000 (every 4 months) via Direct Benefit Transfer (DBT).",
            "eligibility": "All landholding farmer families having cultivable land registered in their name.",
            "exclusions": "Institutional landholders, government employees (excluding Group D/MTS), income tax payees, constitutional post holders, pensioners > ₹10,000/month.",
            "mandatory_checklist": [
                "Aadhaar-linked active bank account with NPCI mapping",
                "Completed e-KYC (via OTP on pmkisan.gov.in or biometric CSC)",
                "Verified land record seeding (Bhulekh / Khatauni / Patta)"
            ],
            "portal": "https://pmkisan.gov.in",
            "helpline": "155261 / 011-24300606"
        },
        "PMFBY": {
            "name": "Pradhan Mantri Fasal Bima Yojana (Crop Insurance)",
            "benefit": "Comprehensive insurance against drought, flood, hailstorm, pest attack, and post-harvest unseasonal rain losses.",
            "farmer_premium": "2.0% for Kharif crops; 1.5% for Rabi crops; 5.0% for Annual Commercial & Horticultural crops.",
            "critical_rule": "Report localized crop damage or post-harvest loss within STRICT 72 HOURS to qualify for claim survey.",
            "mandatory_checklist": [
                "Sowing certificate / land record (7/12, RTC, Adangal)",
                "Bank passbook copy & Aadhaar card",
                "Crop Insurance Policy Number / Application receipt"
            ],
            "portal": "https://pmfby.gov.in",
            "helpline": "1800-180-1551 (Kisan Call Center) or NCIP Crop Insurance App"
        },
        "KCC": {
            "name": "Kisan Credit Card Scheme",
            "benefit": "Subsidized crop production loan up to ₹3 Lakh at an effective interest rate of only 4.0% p.a. (with prompt repayment within 1 year). Collateral-free limit up to ₹1.6 Lakh.",
            "eligibility": "Individual farmers, joint borrowers, tenant farmers, oral lessees, sharecroppers, and SHGs.",
            "mandatory_checklist": [
                "Land ownership document / cultivation tenancy certificate",
                "Aadhaar & PAN card",
                "Passport size photograph & crop cultivation plan"
            ],
            "portal": "Apply at any Nationalized/Cooperative/RRB bank branch",
            "helpline": "1800-180-1551"
        }
    }

    def check_eligibility(self, is_landowner: bool, is_govt_employee: bool, pays_income_tax: bool) -> Dict[str, Any]:
        pm_kisan_eligible = is_landowner and not is_govt_employee and not pays_income_tax
        kcc_eligible = True # Even tenant farmers / sharecroppers eligible
        pmfby_eligible = True # All cultivating farmers eligible

        reasons = []
        if is_govt_employee:
            reasons.append("Excluded from PM-KISAN due to government employment status.")
        if pays_income_tax:
            reasons.append("Excluded from PM-KISAN due to income tax payee status.")
        if not is_landowner:
            reasons.append("PM-KISAN requires registered land ownership in beneficiary name.")

        return {
            "pm_kisan_eligible": pm_kisan_eligible,
            "pmfby_eligible": pmfby_eligible,
            "kcc_eligible": kcc_eligible,
            "pm_kisan_notes": "Eligible for ₹6,000/year" if pm_kisan_eligible else f"Ineligible: {', '.join(reasons)}",
            "schemes_details": self.SCHEMES
        }
