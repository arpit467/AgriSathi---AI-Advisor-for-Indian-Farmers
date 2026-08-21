"""
AgriSathi Smart Fertilizer & Yield ROI Calculator Engine
Calculates land-specific NPK dosages, bag requirements, subsidized input costs, projected yield, and MSP revenue.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field

# Unit conversion to Standard Acres
UNIT_CONVERSION = {
    "acre": 1.0,
    "hectare": 2.47105,
    "bigha": 0.625,  # Standard Northern Bigha (1 Acre = 1.6 Bigha)
}

# Subsidized Fertilizer Prices in India (2024 Rates)
FERTILIZER_PRICES = {
    "urea": {"name": "Urea (46% N)", "bag_weight_kg": 45, "bag_price_inr": 266.50, "cost_per_kg": 5.92},
    "dap": {"name": "DAP (18:46:0)", "bag_weight_kg": 50, "bag_price_inr": 1350.00, "cost_per_kg": 27.00},
    "mop": {"name": "MOP (60% K2O)", "bag_weight_kg": 50, "bag_price_inr": 1700.00, "cost_per_kg": 34.00},
    "zinc": {"name": "Zinc Sulphate (21% Zn)", "bag_weight_kg": 10, "bag_price_inr": 600.00, "cost_per_kg": 60.00},
}

# Crop Agricultural Benchmarks (Per Acre) — Official ICAR & Mandi Benchmarks
CROP_DATA = {
    # ── Cereals & Millets ──
    "Wheat": {
        "dap_kg_per_acre": 50.0, "urea_kg_per_acre": 60.0, "mop_kg_per_acre": 20.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 18.5, "msp_per_quintal_inr": 2275.0,
    },
    "Paddy (Rice)": {
        "dap_kg_per_acre": 40.0, "urea_kg_per_acre": 75.0, "mop_kg_per_acre": 25.0, "zinc_kg_per_acre": 12.0,
        "avg_yield_quintal_per_acre": 22.0, "msp_per_quintal_inr": 2183.0,
    },
    "Corn (Maize)": {
        "dap_kg_per_acre": 45.0, "urea_kg_per_acre": 70.0, "mop_kg_per_acre": 20.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 20.0, "msp_per_quintal_inr": 2090.0,
    },
    "Bajra (Pearl Millet)": {
        "dap_kg_per_acre": 30.0, "urea_kg_per_acre": 40.0, "mop_kg_per_acre": 15.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 12.0, "msp_per_quintal_inr": 2500.0,
    },
    "Jowar (Sorghum)": {
        "dap_kg_per_acre": 35.0, "urea_kg_per_acre": 45.0, "mop_kg_per_acre": 15.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 11.5, "msp_per_quintal_inr": 3180.0,
    },
    "Barley": {
        "dap_kg_per_acre": 35.0, "urea_kg_per_acre": 40.0, "mop_kg_per_acre": 15.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 15.0, "msp_per_quintal_inr": 1850.0,
    },
    "Ragi (Finger Millet)": {
        "dap_kg_per_acre": 25.0, "urea_kg_per_acre": 35.0, "mop_kg_per_acre": 15.0, "zinc_kg_per_acre": 6.0,
        "avg_yield_quintal_per_acre": 10.0, "msp_per_quintal_inr": 3846.0,
    },

    # ── Pulses ──
    "Gram (Chana)": {
        "dap_kg_per_acre": 40.0, "urea_kg_per_acre": 15.0, "mop_kg_per_acre": 15.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 9.5, "msp_per_quintal_inr": 5440.0,
    },
    "Arhar (Tur / Red Gram)": {
        "dap_kg_per_acre": 45.0, "urea_kg_per_acre": 20.0, "mop_kg_per_acre": 15.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 8.0, "msp_per_quintal_inr": 7000.0,
    },
    "Moong (Green Gram)": {
        "dap_kg_per_acre": 35.0, "urea_kg_per_acre": 15.0, "mop_kg_per_acre": 10.0, "zinc_kg_per_acre": 6.0,
        "avg_yield_quintal_per_acre": 6.5, "msp_per_quintal_inr": 8558.0,
    },
    "Urad (Black Gram)": {
        "dap_kg_per_acre": 35.0, "urea_kg_per_acre": 15.0, "mop_kg_per_acre": 10.0, "zinc_kg_per_acre": 6.0,
        "avg_yield_quintal_per_acre": 6.0, "msp_per_quintal_inr": 6950.0,
    },
    "Lentil (Masoor)": {
        "dap_kg_per_acre": 35.0, "urea_kg_per_acre": 15.0, "mop_kg_per_acre": 10.0, "zinc_kg_per_acre": 6.0,
        "avg_yield_quintal_per_acre": 7.5, "msp_per_quintal_inr": 6425.0,
    },

    # ── Oilseeds ──
    "Mustard (Sarson)": {
        "dap_kg_per_acre": 35.0, "urea_kg_per_acre": 45.0, "mop_kg_per_acre": 15.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 8.5, "msp_per_quintal_inr": 5650.0,
    },
    "Groundnut (Peanut)": {
        "dap_kg_per_acre": 40.0, "urea_kg_per_acre": 20.0, "mop_kg_per_acre": 25.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 11.0, "msp_per_quintal_inr": 6377.0,
    },
    "Soybean": {
        "dap_kg_per_acre": 45.0, "urea_kg_per_acre": 20.0, "mop_kg_per_acre": 20.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 9.0, "msp_per_quintal_inr": 4600.0,
    },
    "Sunflower": {
        "dap_kg_per_acre": 40.0, "urea_kg_per_acre": 35.0, "mop_kg_per_acre": 20.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 7.5, "msp_per_quintal_inr": 6760.0,
    },
    "Sesame (Til)": {
        "dap_kg_per_acre": 25.0, "urea_kg_per_acre": 25.0, "mop_kg_per_acre": 10.0, "zinc_kg_per_acre": 5.0,
        "avg_yield_quintal_per_acre": 4.5, "msp_per_quintal_inr": 8635.0,
    },

    # ── Commercial & Cash Crops ──
    "Cotton": {
        "dap_kg_per_acre": 45.0, "urea_kg_per_acre": 90.0, "mop_kg_per_acre": 30.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 10.5, "msp_per_quintal_inr": 6620.0,
    },
    "Sugarcane": {
        "dap_kg_per_acre": 100.0, "urea_kg_per_acre": 150.0, "mop_kg_per_acre": 60.0, "zinc_kg_per_acre": 15.0,
        "avg_yield_quintal_per_acre": 350.0, "msp_per_quintal_inr": 315.0,
    },
    "Jute": {
        "dap_kg_per_acre": 35.0, "urea_kg_per_acre": 50.0, "mop_kg_per_acre": 20.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 13.0, "msp_per_quintal_inr": 5050.0,
    },

    # ── Vegetables & Spices ──
    "Potato": {
        "dap_kg_per_acre": 80.0, "urea_kg_per_acre": 100.0, "mop_kg_per_acre": 50.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 120.0, "msp_per_quintal_inr": 1250.0,
    },
    "Tomato": {
        "dap_kg_per_acre": 60.0, "urea_kg_per_acre": 80.0, "mop_kg_per_acre": 40.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 110.0, "msp_per_quintal_inr": 1400.0,
    },
    "Onion": {
        "dap_kg_per_acre": 55.0, "urea_kg_per_acre": 75.0, "mop_kg_per_acre": 35.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 100.0, "msp_per_quintal_inr": 1650.0,
    },
    "Garlic": {
        "dap_kg_per_acre": 60.0, "urea_kg_per_acre": 80.0, "mop_kg_per_acre": 40.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 35.0, "msp_per_quintal_inr": 4500.0,
    },
    "Ginger": {
        "dap_kg_per_acre": 70.0, "urea_kg_per_acre": 90.0, "mop_kg_per_acre": 45.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 60.0, "msp_per_quintal_inr": 3800.0,
    },
    "Brinjal (Eggplant)": {
        "dap_kg_per_acre": 50.0, "urea_kg_per_acre": 70.0, "mop_kg_per_acre": 30.0, "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 90.0, "msp_per_quintal_inr": 1350.0,
    },
    "Chilli (Red/Green)": {
        "dap_kg_per_acre": 60.0, "urea_kg_per_acre": 85.0, "mop_kg_per_acre": 40.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 25.0, "msp_per_quintal_inr": 7200.0,
    },
    "Okra (Bhindi)": {
        "dap_kg_per_acre": 40.0, "urea_kg_per_acre": 50.0, "mop_kg_per_acre": 25.0, "zinc_kg_per_acre": 6.0,
        "avg_yield_quintal_per_acre": 45.0, "msp_per_quintal_inr": 1500.0,
    },

    # ── Fruits & Plantation ──
    "Apple": {
        "dap_kg_per_acre": 75.0, "urea_kg_per_acre": 90.0, "mop_kg_per_acre": 50.0, "zinc_kg_per_acre": 12.0,
        "avg_yield_quintal_per_acre": 65.0, "msp_per_quintal_inr": 4800.0,
    },
    "Mango": {
        "dap_kg_per_acre": 80.0, "urea_kg_per_acre": 110.0, "mop_kg_per_acre": 60.0, "zinc_kg_per_acre": 15.0,
        "avg_yield_quintal_per_acre": 50.0, "msp_per_quintal_inr": 3200.0,
    },
    "Banana": {
        "dap_kg_per_acre": 110.0, "urea_kg_per_acre": 160.0, "mop_kg_per_acre": 140.0, "zinc_kg_per_acre": 15.0,
        "avg_yield_quintal_per_acre": 240.0, "msp_per_quintal_inr": 1550.0,
    },
    "Citrus (Orange/Lemon)": {
        "dap_kg_per_acre": 70.0, "urea_kg_per_acre": 95.0, "mop_kg_per_acre": 45.0, "zinc_kg_per_acre": 12.0,
        "avg_yield_quintal_per_acre": 55.0, "msp_per_quintal_inr": 3100.0,
    },
    "Guava": {
        "dap_kg_per_acre": 50.0, "urea_kg_per_acre": 70.0, "mop_kg_per_acre": 35.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 45.0, "msp_per_quintal_inr": 2200.0,
    },
    "Papaya": {
        "dap_kg_per_acre": 85.0, "urea_kg_per_acre": 120.0, "mop_kg_per_acre": 80.0, "zinc_kg_per_acre": 12.0,
        "avg_yield_quintal_per_acre": 180.0, "msp_per_quintal_inr": 1400.0,
    },
    "Grape": {
        "dap_kg_per_acre": 90.0, "urea_kg_per_acre": 110.0, "mop_kg_per_acre": 90.0, "zinc_kg_per_acre": 15.0,
        "avg_yield_quintal_per_acre": 90.0, "msp_per_quintal_inr": 4200.0,
    },
    "Pear": {
        "dap_kg_per_acre": 65.0, "urea_kg_per_acre": 80.0, "mop_kg_per_acre": 45.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 55.0, "msp_per_quintal_inr": 3600.0,
    },
    "Plum": {
        "dap_kg_per_acre": 60.0, "urea_kg_per_acre": 75.0, "mop_kg_per_acre": 40.0, "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 40.0, "msp_per_quintal_inr": 3500.0,
    }
}

# Soil Adjustments Multipliers
SOIL_MULTIPLIERS = {
    "Alluvial": {"dap": 1.0, "urea": 1.0, "mop": 1.0, "zinc": 1.0},
    "Black":    {"dap": 0.9, "urea": 1.0, "mop": 0.85, "zinc": 1.1},
    "Clay":     {"dap": 1.05, "urea": 1.1, "mop": 0.95, "zinc": 1.0},
    "Sandy":    {"dap": 1.15, "urea": 1.2, "mop": 1.15, "zinc": 1.25},
    "Loamy":    {"dap": 0.95, "urea": 0.95, "mop": 0.95, "zinc": 0.95},
}


class CalculatorRequest(BaseModel):
    land_size: float = Field(gt=0, default=2.5, description="Land area")
    unit: str = Field(default="acre", description="acre | hectare | bigha")
    crop: str = Field(default="Wheat", description="Wheat | Paddy | Cotton | Mustard | Sugarcane | Potato | Tomato")
    soil_type: str = Field(default="Alluvial", description="Alluvial | Black | Clay | Sandy | Loamy")


class FertilizerBagDetail(BaseModel):
    name: str
    kg_required: float
    bags_required: float
    bag_weight_kg: float
    unit_price_inr: float
    total_cost_inr: float


class CalculatorResponse(BaseModel):
    land_size_input: float
    land_unit_input: str
    equivalent_acres: float
    crop: str
    soil_type: str
    fertilizers: List[FertilizerBagDetail]
    total_fertilizer_cost_inr: float
    estimated_yield_quintals: float
    msp_per_quintal_inr: float
    estimated_gross_revenue_inr: float
    estimated_net_profit_inr: float
    roi_percentage: float


class AgriCalculatorEngine:
    @staticmethod
    def calculate(req: CalculatorRequest) -> CalculatorResponse:
        unit_key = req.unit.lower()
        multiplier = UNIT_CONVERSION.get(unit_key, 1.0)
        acres = round(req.land_size * multiplier, 2)

        crop_info = CROP_DATA.get(req.crop, CROP_DATA["Wheat"])
        soil_mult = SOIL_MULTIPLIERS.get(req.soil_type, SOIL_MULTIPLIERS["Alluvial"])

        # Calculate exact requirements
        dap_kg  = round(crop_info["dap_kg_per_acre"] * acres * soil_mult["dap"], 1)
        urea_kg = round(crop_info["urea_kg_per_acre"] * acres * soil_mult["urea"], 1)
        mop_kg  = round(crop_info["mop_kg_per_acre"] * acres * soil_mult["mop"], 1)
        zinc_kg = round(crop_info["zinc_kg_per_acre"] * acres * soil_mult["zinc"], 1)

        fert_configs = [
            ("dap", dap_kg),
            ("urea", urea_kg),
            ("mop", mop_kg),
            ("zinc", zinc_kg),
        ]

        fertilizer_details = []
        total_fert_cost = 0.0

        for key, kg in fert_configs:
            info = FERTILIZER_PRICES[key]
            bags = round(kg / info["bag_weight_kg"], 1)
            cost = round(kg * info["cost_per_kg"], 2)
            total_fert_cost += cost

            fertilizer_details.append(FertilizerBagDetail(
                name=info["name"],
                kg_required=kg,
                bags_required=bags,
                bag_weight_kg=info["bag_weight_kg"],
                unit_price_inr=info["bag_price_inr"],
                total_cost_inr=cost
            ))

        total_fert_cost = round(total_fert_cost, 2)

        # Economic Yield & MSP Revenue calculations
        yield_quintals = round(crop_info["avg_yield_quintal_per_acre"] * acres, 1)
        msp_rate = crop_info["msp_per_quintal_inr"]
        gross_revenue  = round(yield_quintals * msp_rate, 2)
        
        # Estimate total cultivation cost (fertilizer + labor/seeds/irrigation ~ 2.5x fert cost)
        total_estimated_cost = round(total_fert_cost * 2.5, 2)
        net_profit = round(gross_revenue - total_estimated_cost, 2)
        roi_pct = round((net_profit / max(1, total_estimated_cost)) * 100, 1)

        return CalculatorResponse(
            land_size_input=req.land_size,
            land_unit_input=req.unit,
            equivalent_acres=acres,
            crop=req.crop,
            soil_type=req.soil_type,
            fertilizers=fertilizer_details,
            total_fertilizer_cost_inr=total_fert_cost,
            estimated_yield_quintals=yield_quintals,
            msp_per_quintal_inr=msp_rate,
            estimated_gross_revenue_inr=gross_revenue,
            estimated_net_profit_inr=net_profit,
            roi_percentage=roi_pct
        )
