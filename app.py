from pathlib import Path
from typing import Any
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download 


# Chemins robustes, indépendants du dossier depuis lequel uvicorn est lancé
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


app = FastAPI(
    title="AI Travel Planner API",
    description=(
        "API de prédiction du prix des vols "
        "et de la fiabilité des réservations d'hôtel."
    ),
    version="1.1.0"
)


# Dépôt Hugging Face contenant les modèles et les encodeurs
REPO_ID = "Emilie7/ai-travel-planner-models"


def get_model_path(filename: str) -> Path:
    """
    Utilise le fichier local s'il existe.
    Sinon, le télécharge depuis Hugging Face.
    """
    local_path = MODELS_DIR / filename

    if local_path.exists():
        print(f"Chargement local : {filename}")
        return local_path

    print(f"Téléchargement depuis Hugging Face : {filename}")

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename
    )

    return Path(downloaded_path)


# Chargement des modèles
hotel_model = joblib.load(
    get_model_path("hotel_model.pkl")
)

flight_model = joblib.load(
    get_model_path("flight_model.pkl")
)

# Chargement des encodeurs
hotel_encoders = joblib.load(
    get_model_path("hotel_encoders.pkl")
)

flight_encoders = joblib.load(
    get_model_path("flight_encoders.pkl")
)


with open(
    DATA_DIR / "destinations.json",
    "r",
    encoding="utf-8"
) as file:
    DESTINATIONS = json.load(file)


def encode_value(
    encoders: dict[str, Any],
    column: str,
    value: str
) -> int:
    """
    Encode une valeur texte avec le LabelEncoder associé à la colonne.
    Renvoie une erreur 422 si la valeur n'existe pas dans les données.
    """
    if column not in encoders:
        raise HTTPException(
            status_code=500,
            detail=f"Aucun encodeur trouvé pour la colonne '{column}'."
        )

    encoder = encoders[column]
    cleaned_value = value.strip()

    valid_values = [str(item) for item in encoder.classes_]

    if cleaned_value not in valid_values:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"Valeur '{cleaned_value}' inconnue "
                    f"pour le champ '{column}'."
                ),
                "allowed_values": valid_values
            }
        )

    return int(encoder.transform([cleaned_value])[0])


@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API AI Travel Planner",
        "version": "1.1.0"
    }


class HotelRequest(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: float
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int


class FlightRequest(BaseModel):
    airline: str
    flight: str
    source_city: str
    departure_time: str
    stops: str
    arrival_time: str
    destination_city: str
    travel_class: str
    duration: float
    days_left: int


class BudgetRequest(BaseModel):
    flight_price_per_person: float = Field(gt=0)
    accommodation_price_per_night: float = Field(ge=0)
    nights: int = Field(ge=0)
    trip_days: int = Field(ge=1)
    travelers: int = Field(default=1, ge=1)
    daily_expenses_per_person: float = Field(ge=0)
    local_transport: float = Field(default=0, ge=0)
    activities: float = Field(default=0, ge=0)
    contingency_rate: float = Field(default=0.10, ge=0, le=0.50)
    currency: str = "INR"    


class RecommendationRequest(BaseModel):
    budget_per_person: float = Field(gt=0)
    trip_days: int = Field(ge=1, le=30)
    preferred_climate: str
    travel_style: str
    preferred_activity: str
    source_city: str | None = None


@app.post("/predict-hotel")
def predict_hotel(data: HotelRequest):
    input_data = data.model_dump()

    # Feature engineering identique au notebook 02
    input_data["total_nights"] = (
        input_data["stays_in_weekend_nights"]
        + input_data["stays_in_week_nights"]
    )

    input_data["total_guests"] = (
        input_data["adults"]
        + input_data["children"]
        + input_data["babies"]
    )

    # Colonnes catégorielles encodées dans le notebook 02
    categorical_fields = [
        "hotel",
        "arrival_date_month",
        "meal",
        "country",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "assigned_room_type",
        "deposit_type",
        "customer_type"
    ]

    for field in categorical_fields:
        input_data[field] = encode_value(
            hotel_encoders,
            field,
            input_data[field]
        )

    columns = [
        "hotel",
        "lead_time",
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "meal",
        "country",
        "market_segment",
        "distribution_channel",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "reserved_room_type",
        "assigned_room_type",
        "booking_changes",
        "deposit_type",
        "agent",
        "days_in_waiting_list",
        "customer_type",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "total_nights",
        "total_guests"
    ]

    df = pd.DataFrame([input_data], columns=columns)

    prediction = int(hotel_model.predict(df)[0])
    probability = float(hotel_model.predict_proba(df)[0][1])

    if probability < 0.30:
        reliability = "Fiabilité élevée"
        cancellation_risk = "Risque faible d'annulation"

    elif probability < 0.50:
        reliability = "Fiabilité moyenne"
        cancellation_risk = "Risque modéré d'annulation"

    else:
        reliability = "Fiabilité faible"
        cancellation_risk = "Risque élevé d'annulation"

    return {
        "reservation_reliability": reliability,
        "cancellation_risk": cancellation_risk,
        "prediction": prediction,
        "cancellation_probability": round(probability, 4),
        "cancellation_probability_percent": round(probability * 100, 2)
    }


@app.post("/predict-flight")
def predict_flight(data: FlightRequest):
    input_data = data.model_dump()

    # Le dataset utilise la colonne "class"
    input_data["class"] = input_data.pop("travel_class")

    categorical_fields = [
        "airline",
        "flight",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class"
    ]

    for field in categorical_fields:
        input_data[field] = encode_value(
            flight_encoders,
            field,
            input_data[field]
        )

    columns = [
        "airline",
        "flight",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
        "duration",
        "days_left"
    ]

    df = pd.DataFrame([input_data], columns=columns)

    prediction = float(flight_model.predict(df)[0])

    return {
        "predicted_price": round(prediction, 2),
        "currency": "INR"
    }


@app.post("/budget")
def estimate_budget(data: BudgetRequest):
    flight_total = (
        data.flight_price_per_person
        * data.travelers
    )

    accommodation_total = (
        data.accommodation_price_per_night
        * data.nights
    )

    daily_expenses_total = (
        data.daily_expenses_per_person
        * data.trip_days
        * data.travelers
    )

    subtotal = (
        flight_total
        + accommodation_total
        + daily_expenses_total
        + data.local_transport
        + data.activities
    )

    contingency_amount = subtotal * data.contingency_rate
    total_budget = subtotal + contingency_amount

    return {
        "estimated_budget": round(total_budget, 2),
        "budget_per_person": round(
            total_budget / data.travelers,
            2
        ),
        "currency": data.currency.strip().upper(),
        "breakdown": {
            "flights": round(flight_total, 2),
            "accommodation": round(accommodation_total, 2),
            "daily_expenses": round(daily_expenses_total, 2),
            "local_transport": round(data.local_transport, 2),
            "activities": round(data.activities, 2),
            "contingency": round(contingency_amount, 2)
        }
    }


@app.post("/recommendation")
def recommend_destination(data: RecommendationRequest):
    climate = data.preferred_climate.strip().lower()
    travel_style = data.travel_style.strip().lower()
    activity = data.preferred_activity.strip().lower()

    source_city = (
        data.source_city.strip().lower()
        if data.source_city
        else None
    )

    recommendations = []

    for destination in DESTINATIONS:
        destination_name = destination["destination"]

        # Évite de recommander la ville de départ
        if (
            source_city
            and destination_name.lower() == source_city
        ):
            continue

        score = 0
        reasons = []

        if destination["climate"] == climate:
            score += 3
            reasons.append("Climat correspondant")

        if travel_style in destination["styles"]:
            score += 2
            reasons.append("Style de voyage correspondant")

        if activity in destination["activities"]:
            score += 2
            reasons.append("Activité recherchée disponible")

        estimated_local_cost = (
            destination["daily_cost"]
            * data.trip_days
        )

        within_budget = (
            estimated_local_cost
            <= data.budget_per_person
        )

        if within_budget:
            score += 2
            reasons.append("Compatible avec le budget")
        else:
            score -= 2
            reasons.append("Budget insuffisant")

        recommendations.append({
            "destination": destination_name,
            "match_score": score,
            "estimated_local_cost": round(
                estimated_local_cost,
                2
            ),
            "within_budget": within_budget,
            "reasons": reasons
        })

    recommendations.sort(
        key=lambda item: (
            item["within_budget"],
            item["match_score"],
            -item["estimated_local_cost"]
        ),
        reverse=True
    )

    return {
        "recommendations": recommendations[:3],
        "currency": "INR",
        "method": "rule_based"
    }