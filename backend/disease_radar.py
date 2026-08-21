"""
AgriSathi AI Crop Disease Outbreak Prediction Radar Engine
Predicts fungal, bacterial, and pest outbreak probabilities based on microclimate sensor inputs (temperature, humidity, rainfall).
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class RadarPredictionRequest(BaseModel):
    crop: str = Field(default="Wheat", description="Wheat | Paddy | Potato | Cotton | Mustard | Tomato")
    temperature_c: float = Field(default=15.0, description="Ambient temperature in °C")
    humidity_pct: float = Field(default=88.0, description="Relative humidity percentage")
    rain_forecast_mm: float = Field(default=12.0, description="Rainfall forecast in mm over 48h")


class RadarPredictionResponse(BaseModel):
    crop: str
    temperature_c: float
    humidity_pct: float
    rain_forecast_mm: float
    primary_disease: str
    outbreak_probability_pct: float
    risk_level: str  # "CRITICAL" | "MODERATE" | "LOW"
    risk_color: str
    contributing_factors: List[str]
    preventive_protocol: List[str]
    action_recommendation: str


# Microclimate Disease Trigger Rules Matrix
DISEASE_RULES = {
    "Wheat": {
        "disease": "Yellow Rust (Puccinia striiformis)",
        "ideal_temp": (10.0, 18.0),
        "min_humidity": 80.0,
        "preventive": [
            "Postpone nitrogen fertilizer top-dressing until weather clears.",
            "Spray bio-fungicide (Trichoderma viride @ 5g/L) as a preventive shield.",
            "If humidity > 85% persists for 48h, apply Propiconazole 25% EC @ 1.0 ml/L before pustule formation.",
            "Ensure proper field drainage to prevent localized high micro-humidity."
        ]
    },
    "Paddy": {
        "disease": "Rice Blast & Sheath Blight (Pyricularia oryzae)",
        "ideal_temp": (22.0, 29.0),
        "min_humidity": 85.0,
        "preventive": [
            "Avoid excessive nitrogenous fertilizer application during humid spells.",
            "Spray Tricyclazole 75% WP @ 0.6 g/L or Validamycin 3% L @ 2 ml/L.",
            "Drain excess water from paddies periodically to reduce humidity build-up."
        ]
    },
    "Potato": {
        "disease": "Late Blight (Phytophthora infestans)",
        "ideal_temp": (12.0, 20.0),
        "min_humidity": 88.0,
        "preventive": [
            "Prophylactic spray of Mancozeb 75% WP @ 2.0 g/L of water.",
            "Destroy infected foliage (earthing up) if initial lesions appear.",
            "Avoid sprinkler irrigation during cool foggy mornings."
        ]
    },
    "Cotton": {
        "disease": "Pink Bollworm & Bacterial Blight",
        "ideal_temp": (28.0, 35.0),
        "min_humidity": 65.0,
        "preventive": [
            "Install Pheromone traps @ 5 traps per acre for early moth monitoring.",
            "Spray Emamectin Benzoate 5% SG @ 4g / 10L water.",
            "Destroy crop residues and infested bolls after picking."
        ]
    },
    "Mustard": {
        "disease": "White Rust & Alternaria Blight",
        "ideal_temp": (12.0, 22.0),
        "min_humidity": 75.0,
        "preventive": [
            "Spray Metalaxyl 8% + Mancozeb 64% WP @ 2g/L of water.",
            "Maintain optimal plant spacing for proper airflow."
        ]
    },
    "Tomato": {
        "disease": "Early & Late Blight / Leaf Curl Virus",
        "ideal_temp": (18.0, 26.0),
        "min_humidity": 80.0,
        "preventive": [
            "Spray Copper Oxychloride 50% WP @ 3.0 g/L.",
            "Use yellow sticky traps @ 10 per acre to catch vector whiteflies."
        ]
    }
}


class DiseaseOutbreakRadarEngine:
    @staticmethod
    def predict(req: RadarPredictionRequest) -> RadarPredictionResponse:
        rule = DISEASE_RULES.get(req.crop, DISEASE_RULES["Wheat"])
        
        factors = []
        prob_score = 30.0  # Base risk

        # Temperature Factor
        min_t, max_t = rule["ideal_temp"]
        if min_t <= req.temperature_c <= max_t:
            prob_score += 35.0
            factors.append(f"Temperature ({req.temperature_c}°C) is in optimal fungal spore germination range ({min_t}-{max_t}°C).")
        elif abs(req.temperature_c - (min_t + max_t)/2) < 5.0:
            prob_score += 15.0
            factors.append(f"Temperature ({req.temperature_c}°C) near disease threshold.")

        # Humidity Factor
        if req.humidity_pct >= rule["min_humidity"]:
            prob_score += 25.0
            factors.append(f"High Relative Humidity ({req.humidity_pct}%) exceeds critical threshold ({rule['min_humidity']}%).")
        elif req.humidity_pct >= 70.0:
            prob_score += 10.0

        # Rainfall Factor
        if req.rain_forecast_mm > 5.0:
            prob_score += 10.0
            factors.append(f"48-hour Rainfall forecast ({req.rain_forecast_mm} mm) accelerates foliar moisture.")

        final_prob = min(98.0, round(prob_score, 1))

        if final_prob >= 80.0:
            risk_level = "CRITICAL"
            color = "#FF3366"
            action = f"⚠️ CRITICAL WARNING: High risk of {rule['disease']}. Execute preventive spray protocol within 24-48 hours."
        elif final_prob >= 55.0:
            risk_level = "MODERATE"
            color = "#FF9F1C"
            action = f"⚡ MODERATE RISK: Fungal spores active for {rule['disease']}. Monitor field daily and prepare protective bio-pesticide."
        else:
            risk_level = "LOW"
            color = "#00F5A0"
            action = f"✅ LOW RISK: Microclimate conditions are stable for {req.crop}."

        return RadarPredictionResponse(
            crop=req.crop,
            temperature_c=req.temperature_c,
            humidity_pct=req.humidity_pct,
            rain_forecast_mm=req.rain_forecast_mm,
            primary_disease=rule["disease"],
            outbreak_probability_pct=final_prob,
            risk_level=risk_level,
            risk_color=color,
            contributing_factors=factors if factors else ["Microclimate parameters within safe limits."],
            preventive_protocol=rule["preventive"],
            action_recommendation=action
        )
