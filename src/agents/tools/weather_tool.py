from typing import Dict, Any

class AgroWeatherAdvisor:
    """
    Agricultural Weather & Spray Advisory Tool.
    Evaluates weather parameters (real or localized farmer input) to provide
    actionable operational advice regarding pesticide spraying, irrigation, and frost protection.
    """
    def evaluate_spray_conditions(self, temp_c: float, humidity_pct: float, wind_kmh: float, rain_prob_pct: float) -> Dict[str, Any]:
        warnings = []
        spray_verdict = "FAVORABLE"
        color = "green"

        # Wind speed rule
        if wind_kmh > 15.0:
            warnings.append(f"High wind speed ({wind_kmh} km/h) creates excessive spray drift risk. Do not spray herbicides or toxic insecticides.")
            spray_verdict = "UNFAVORABLE"
            color = "red"
        elif wind_kmh < 3.0:
            warnings.append(f"Very low wind ({wind_kmh} km/h) with thermal inversions may cause uneven spray deposition.")

        # Rain rule
        if rain_prob_pct > 50.0:
            warnings.append(f"High rain probability ({rain_prob_pct}%). Chemical wash-off likely within rain-fastness window (6 hours). Postpone foliar spraying.")
            spray_verdict = "UNFAVORABLE"
            color = "red"
        elif rain_prob_pct > 30.0:
            warnings.append("Moderate rain probability. Use an agricultural sticker/spreader adjuvant if spraying is urgent.")

        # Temperature & Humidity rule
        if temp_c > 35.0:
            warnings.append(f"High temperature ({temp_c}°C) increases droplet evaporation rate and risk of chemical scorching (phytotoxicity). Spray only during early morning or late evening.")
        
        if humidity_pct > 80.0 and (20.0 <= temp_c <= 30.0):
            warnings.append("High humidity and warm conditions favor rapid fungal spore germination (Blight/Blast risk). Inspect fields closely.")

        # Frost rule for North India Rabi
        if temp_c <= 4.0:
            warnings.append("Frost/Cold wave risk: Apply light evening irrigation to raise soil temperature and shield standing Rabi crops.")

        return {
            "spray_verdict": spray_verdict,
            "status_color": color,
            "weather_input": {
                "Temperature": f"{temp_c} °C",
                "Relative Humidity": f"{humidity_pct} %",
                "Wind Speed": f"{wind_kmh} km/h",
                "Rain Probability": f"{rain_prob_pct} %"
            },
            "advisory_warnings": warnings or ["Weather conditions are optimal for foliar spray and field operations."],
            "operational_tips": [
                "Always use flat fan nozzles for herbicide sprays and hollow cone for insecticide/fungicide sprays.",
                "Maintain nozzle height 40-50 cm above crop canopy.",
                "Avoid spraying between 12:00 PM and 3:00 PM under direct hot sun."
            ]
        }

