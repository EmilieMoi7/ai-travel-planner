from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class FakeEncoder:
    def __init__(self, values):
        self.classes_ = values

    def transform(self, values):
        return [list(self.classes_).index(value) for value in values]


class FakeHotelModel:
    def predict(self, dataframe):
        return [0]

    def predict_proba(self, dataframe):
        return [[0.60, 0.40]]


class FakeFlightModel:
    def predict(self, dataframe):
        return [2620.76]


HOTEL_ENCODERS = {
    "hotel": FakeEncoder(["City Hotel", "Resort Hotel"]),
    "arrival_date_month": FakeEncoder(["July"]),
    "meal": FakeEncoder(["BB"]),
    "country": FakeEncoder(["PRT"]),
    "market_segment": FakeEncoder(["Online TA"]),
    "distribution_channel": FakeEncoder(["TA/TO"]),
    "reserved_room_type": FakeEncoder(["A"]),
    "assigned_room_type": FakeEncoder(["A"]),
    "deposit_type": FakeEncoder(["No Deposit"]),
    "customer_type": FakeEncoder(["Transient"]),
}

FLIGHT_ENCODERS = {
    "airline": FakeEncoder(["SpiceJet"]),
    "flight": FakeEncoder(["SG-8709"]),
    "source_city": FakeEncoder(["Delhi"]),
    "departure_time": FakeEncoder(["Evening"]),
    "stops": FakeEncoder(["zero"]),
    "arrival_time": FakeEncoder(["Night"]),
    "destination_city": FakeEncoder(["Mumbai"]),
    "class": FakeEncoder(["Economy"]),
}


def fake_joblib_load(path):
    filename = Path(path).name

    objects = {
        "hotel_model.pkl": FakeHotelModel(),
        "flight_model.pkl": FakeFlightModel(),
        "hotel_encoders.pkl": HOTEL_ENCODERS,
        "flight_encoders.pkl": FLIGHT_ENCODERS,
    }

    return objects[filename]


def fake_hf_hub_download(repo_id, filename):
    return f"/tmp/{filename}"


# Les modèles sont simulés avant l'import de l'application.
with (
    patch("joblib.load", side_effect=fake_joblib_load),
    patch(
        "huggingface_hub.hf_hub_download",
        side_effect=fake_hf_hub_download,
    ),
):
    import app


client = TestClient(app.app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "1.1.0"


def test_predict_hotel():
    payload = {
        "hotel": "City Hotel",
        "lead_time": 30,
        "arrival_date_year": 2017,
        "arrival_date_month": "July",
        "arrival_date_week_number": 27,
        "arrival_date_day_of_month": 10,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 3,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "agent": 9,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 120,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
    }

    response = client.post("/predict-hotel", json=payload)
    result = response.json()

    assert response.status_code == 200
    assert result["prediction"] == 0
    assert result["reservation_reliability"] == "Fiabilité moyenne"
    assert result["cancellation_probability_percent"] == 40.0


def test_predict_hotel_with_unknown_category():
    payload = {
        "hotel": "Unknown Hotel",
        "lead_time": 30,
        "arrival_date_year": 2017,
        "arrival_date_month": "July",
        "arrival_date_week_number": 27,
        "arrival_date_day_of_month": 10,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 3,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "agent": 9,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 120,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
    }

    response = client.post("/predict-hotel", json=payload)

    assert response.status_code == 422


def test_predict_flight():
    payload = {
        "airline": "SpiceJet",
        "flight": "SG-8709",
        "source_city": "Delhi",
        "departure_time": "Evening",
        "stops": "zero",
        "arrival_time": "Night",
        "destination_city": "Mumbai",
        "travel_class": "Economy",
        "duration": 2.17,
        "days_left": 20,
    }

    response = client.post("/predict-flight", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "predicted_price": 2620.76,
        "currency": "INR",
    }


def test_budget():
    payload = {
        "flight_price_per_person": 2620.76,
        "accommodation_price_per_night": 3500,
        "nights": 4,
        "trip_days": 5,
        "travelers": 2,
        "daily_expenses_per_person": 1200,
        "local_transport": 2500,
        "activities": 4000,
        "contingency_rate": 0.10,
        "currency": "INR",
    }

    response = client.post("/budget", json=payload)
    result = response.json()

    assert response.status_code == 200
    assert result["estimated_budget"] == 41515.67
    assert result["currency"] == "INR"
    assert result["breakdown"]["flights"] == 5241.52


def test_recommendation():
    payload = {
        "budget_per_person": 25000,
        "trip_days": 5,
        "preferred_climate": "warm",
        "travel_style": "culture",
        "preferred_activity": "food",
        "source_city": "Delhi",
    }

    response = client.post("/recommendation", json=payload)
    result = response.json()

    assert response.status_code == 200
    assert result["method"] == "rule_based"
    assert len(result["recommendations"]) == 3
    assert all(
        item["destination"] != "Delhi"
        for item in result["recommendations"]
    )