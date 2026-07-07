from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Chemins robustes, indépendants du dossier depuis lequel uvicorn est lancé
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


app = FastAPI(
    title="AI Travel Planner API",
    description=(
        "API de prédiction du prix des vols "
        "et de la fiabilité des réservations d'hôtel."
    ),
    version="1.1.0"
)


# Chargement des modèles
hotel_model = joblib.load(MODELS_DIR / "hotel_model.pkl")
flight_model = joblib.load(MODELS_DIR / "flight_model.pkl")

# Chargement des encodeurs
hotel_encoders = joblib.load(MODELS_DIR / "hotel_encoders.pkl")
flight_encoders = joblib.load(MODELS_DIR / "flight_encoders.pkl")


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