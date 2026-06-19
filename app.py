from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="AI Travel Planner API",
    description="API de prédiction pour les réservations d'hôtels et les prix des vols.",
    version="1.0.0"
)

hotel_model = joblib.load("models/hotel_model.pkl")
flight_model = joblib.load("models/flight_model.pkl")


@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API AI Travel Planner"}


class HotelRequest(BaseModel):
    hotel: int
    lead_time: int
    arrival_date_year: int
    arrival_date_month: int
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    meal: int
    country: int
    market_segment: int
    distribution_channel: int
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: int
    assigned_room_type: int
    booking_changes: int
    deposit_type: int
    agent: float
    days_in_waiting_list: int
    customer_type: int
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int


@app.post("/predict-hotel")
def predict_hotel(data: HotelRequest):
    input_data = data.model_dump()

    input_data["total_nights"] = (
        input_data["stays_in_weekend_nights"]
        + input_data["stays_in_week_nights"]
    )

    input_data["total_guests"] = (
        input_data["adults"]
        + input_data["children"]
        + input_data["babies"]
    )

    columns = [
        "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
        "arrival_date_week_number", "arrival_date_day_of_month",
        "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies", "meal", "country",
        "market_segment", "distribution_channel", "is_repeated_guest",
        "previous_cancellations", "previous_bookings_not_canceled",
        "reserved_room_type", "assigned_room_type", "booking_changes",
        "deposit_type", "agent", "days_in_waiting_list", "customer_type",
        "adr", "required_car_parking_spaces", "total_of_special_requests",
        "total_nights", "total_guests"
    ]

    df = pd.DataFrame([input_data], columns=columns)

    prediction = hotel_model.predict(df)[0]

    return {
        "prediction": int(prediction),
        "result": "Réservation annulée" if prediction == 1 else "Réservation maintenue"
    }