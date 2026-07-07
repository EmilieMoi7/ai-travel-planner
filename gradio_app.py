from typing import Any

import gradio as gr
from fastapi import HTTPException
from pydantic import ValidationError

from app import (
    DESTINATIONS,
    app as fastapi_app,
    BudgetRequest,
    FlightRequest,
    HotelRequest,
    RecommendationRequest,
    estimate_budget,
    flight_encoders,
    hotel_encoders,
    predict_flight,
    predict_hotel,
    recommend_destination,
)


HOTEL_FIELDS = [
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
]

FLIGHT_FIELDS = [
    "airline",
    "flight",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "travel_class",
    "duration",
    "days_left",
]

BUDGET_FIELDS = [
    "flight_price_per_person",
    "accommodation_price_per_night",
    "nights",
    "trip_days",
    "travelers",
    "daily_expenses_per_person",
    "local_transport",
    "activities",
    "contingency_rate",
    "currency",
]

RECOMMENDATION_FIELDS = [
    "budget_per_person",
    "trip_days",
    "preferred_climate",
    "travel_style",
    "preferred_activity",
    "source_city",
]


def encoder_choices(
    encoders: dict[str, Any],
    field: str,
) -> list[str]:
    return [
        str(value)
        for value in encoders[field].classes_
    ]


def raise_gradio_error(error: Exception) -> None:
    if isinstance(error, HTTPException):
        detail = error.detail
    elif isinstance(error, ValidationError):
        detail = error.errors()
    else:
        detail = str(error)

    raise gr.Error(f"Erreur : {detail}")


def run_hotel_prediction(*values):
    try:
        payload = dict(
            zip(HOTEL_FIELDS, values, strict=True)
        )
        request = HotelRequest(**payload)
        return predict_hotel(request)
    except Exception as error:
        raise_gradio_error(error)


def run_flight_prediction(*values):
    try:
        payload = dict(
            zip(FLIGHT_FIELDS, values, strict=True)
        )
        request = FlightRequest(**payload)
        return predict_flight(request)
    except Exception as error:
        raise_gradio_error(error)


def run_budget_estimation(*values):
    try:
        payload = dict(
            zip(BUDGET_FIELDS, values, strict=True)
        )
        request = BudgetRequest(**payload)
        return estimate_budget(request)
    except Exception as error:
        raise_gradio_error(error)


def run_recommendation(*values):
    try:
        payload = dict(
            zip(RECOMMENDATION_FIELDS, values, strict=True)
        )

        if not payload["source_city"]:
            payload["source_city"] = None

        request = RecommendationRequest(**payload)
        return recommend_destination(request)
    except Exception as error:
        raise_gradio_error(error)


climates = sorted({
    destination["climate"]
    for destination in DESTINATIONS
})

travel_styles = sorted({
    style
    for destination in DESTINATIONS
    for style in destination["styles"]
})

activities = sorted({
    activity
    for destination in DESTINATIONS
    for activity in destination["activities"]
})

cities = encoder_choices(
    flight_encoders,
    "source_city",
)


with gr.Blocks(
    title="AI Travel Planner"
) as demo:
    gr.Markdown(
        """
        # AI Travel Planner

        Prédiction de prix, fiabilité des réservations,
        estimation du budget et recommandation de destination.
        """
    )

    # ---------------------------------------------------------
    # PRÉDICTION DU PRIX D'UN VOL
    # ---------------------------------------------------------

    with gr.Tab("✈️ Prix du vol"):
        with gr.Row():
            airline = gr.Dropdown(
                choices=encoder_choices(
                    flight_encoders,
                    "airline",
                ),
                value="SpiceJet",
                label="Compagnie aérienne",
            )

            flight = gr.Dropdown(
                choices=encoder_choices(
                    flight_encoders,
                    "flight",
                ),
                value="SG-8709",
                label="Numéro de vol",
            )

        with gr.Row():
            source_city = gr.Dropdown(
                choices=cities,
                value="Delhi",
                label="Ville de départ",
            )

            destination_city = gr.Dropdown(
                choices=encoder_choices(
                    flight_encoders,
                    "destination_city",
                ),
                value="Mumbai",
                label="Destination",
            )

        with gr.Row():
            departure_time = gr.Dropdown(
                choices=encoder_choices(
                    flight_encoders,
                    "departure_time",
                ),
                value="Evening",
                label="Moment du départ",
            )

            arrival_time = gr.Dropdown(
                choices=encoder_choices(
                    flight_encoders,
                    "arrival_time",
                ),
                value="Night",
                label="Moment de l'arrivée",
            )

        with gr.Row():
            stops = gr.Dropdown(
                choices=encoder_choices(
                    flight_encoders,
                    "stops",
                ),
                value="zero",
                label="Escales",
            )

            travel_class = gr.Dropdown(
                choices=encoder_choices(
                    flight_encoders,
                    "class",
                ),
                value="Economy",
                label="Classe",
            )

        with gr.Row():
            duration = gr.Number(
                label="Durée du vol",
                value=2.17,
            )

            days_left = gr.Number(
                label="Jours avant le départ",
                value=20,
                precision=0,
            )

        flight_button = gr.Button(
            "Prédire le prix",
            variant="primary",
        )

        flight_output = gr.JSON(
            label="Résultat",
        )

        flight_button.click(
            fn=run_flight_prediction,
            inputs=[
                airline,
                flight,
                source_city,
                departure_time,
                stops,
                arrival_time,
                destination_city,
                travel_class,
                duration,
                days_left,
            ],
            outputs=flight_output,
        )

    # ---------------------------------------------------------
    # FIABILITÉ DE LA RÉSERVATION
    # ---------------------------------------------------------

    with gr.Tab("🏨 Fiabilité de la réservation"):
        with gr.Accordion(
            "Informations principales",
            open=True,
        ):
            with gr.Row():
                hotel = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "hotel",
                    ),
                    value="City Hotel",
                    label="Type d'hôtel",
                )

                lead_time = gr.Number(
                    label="Délai avant l'arrivée",
                    value=30,
                    precision=0,
                )

                adr = gr.Number(
                    label="Prix moyen par nuit",
                    value=120,
                )

        with gr.Accordion(
            "Date et durée du séjour",
            open=True,
        ):
            with gr.Row():
                arrival_date_year = gr.Number(
                    label="Année",
                    value=2017,
                    precision=0,
                )

                arrival_date_month = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "arrival_date_month",
                    ),
                    value="July",
                    label="Mois",
                )

                arrival_date_week_number = gr.Number(
                    label="Numéro de semaine",
                    value=27,
                    precision=0,
                )

                arrival_date_day_of_month = gr.Number(
                    label="Jour du mois",
                    value=10,
                    precision=0,
                )

            with gr.Row():
                stays_in_weekend_nights = gr.Number(
                    label="Nuits le week-end",
                    value=2,
                    precision=0,
                )

                stays_in_week_nights = gr.Number(
                    label="Nuits en semaine",
                    value=3,
                    precision=0,
                )

                days_in_waiting_list = gr.Number(
                    label="Jours en liste d'attente",
                    value=0,
                    precision=0,
                )

        with gr.Accordion("Voyageurs"):
            with gr.Row():
                adults = gr.Number(
                    label="Adultes",
                    value=2,
                    precision=0,
                )

                children = gr.Number(
                    label="Enfants",
                    value=0,
                )

                babies = gr.Number(
                    label="Bébés",
                    value=0,
                    precision=0,
                )

        with gr.Accordion("Réservation"):
            with gr.Row():
                meal = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "meal",
                    ),
                    value="BB",
                    label="Formule repas",
                )

                country = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "country",
                    ),
                    value="PRT",
                    label="Pays",
                )

                market_segment = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "market_segment",
                    ),
                    value="Online TA",
                    label="Segment de marché",
                )

            with gr.Row():
                distribution_channel = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "distribution_channel",
                    ),
                    value="TA/TO",
                    label="Canal de distribution",
                )

                reserved_room_type = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "reserved_room_type",
                    ),
                    value="A",
                    label="Chambre réservée",
                )

                assigned_room_type = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "assigned_room_type",
                    ),
                    value="A",
                    label="Chambre attribuée",
                )

            with gr.Row():
                deposit_type = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "deposit_type",
                    ),
                    value="No Deposit",
                    label="Type de dépôt",
                )

                customer_type = gr.Dropdown(
                    choices=encoder_choices(
                        hotel_encoders,
                        "customer_type",
                    ),
                    value="Transient",
                    label="Type de client",
                )

                agent = gr.Number(
                    label="Identifiant agent",
                    value=9,
                )

        with gr.Accordion("Historique et demandes"):
            with gr.Row():
                is_repeated_guest = gr.Number(
                    label="Client régulier (0 ou 1)",
                    value=0,
                    precision=0,
                )

                previous_cancellations = gr.Number(
                    label="Annulations précédentes",
                    value=0,
                    precision=0,
                )

                previous_bookings_not_canceled = gr.Number(
                    label="Réservations précédentes maintenues",
                    value=0,
                    precision=0,
                )

            with gr.Row():
                booking_changes = gr.Number(
                    label="Modifications de réservation",
                    value=0,
                    precision=0,
                )

                required_car_parking_spaces = gr.Number(
                    label="Places de parking",
                    value=0,
                    precision=0,
                )

                total_of_special_requests = gr.Number(
                    label="Demandes spéciales",
                    value=1,
                    precision=0,
                )

        hotel_button = gr.Button(
            "Vérifier la fiabilité",
            variant="primary",
        )

        hotel_output = gr.JSON(
            label="Résultat",
        )

        hotel_button.click(
            fn=run_hotel_prediction,
            inputs=[
                hotel,
                lead_time,
                arrival_date_year,
                arrival_date_month,
                arrival_date_week_number,
                arrival_date_day_of_month,
                stays_in_weekend_nights,
                stays_in_week_nights,
                adults,
                children,
                babies,
                meal,
                country,
                market_segment,
                distribution_channel,
                is_repeated_guest,
                previous_cancellations,
                previous_bookings_not_canceled,
                reserved_room_type,
                assigned_room_type,
                booking_changes,
                deposit_type,
                agent,
                days_in_waiting_list,
                customer_type,
                adr,
                required_car_parking_spaces,
                total_of_special_requests,
            ],
            outputs=hotel_output,
        )

    # ---------------------------------------------------------
    # ESTIMATION DU BUDGET
    # ---------------------------------------------------------

    with gr.Tab("💰 Estimation du budget"):
        with gr.Row():
            flight_price = gr.Number(
                label="Prix du vol par personne",
                value=2620.76,
            )

            accommodation_price = gr.Number(
                label="Prix de l'hébergement par nuit",
                value=3500,
            )

        with gr.Row():
            nights = gr.Number(
                label="Nombre de nuits",
                value=4,
                precision=0,
            )

            trip_days = gr.Number(
                label="Nombre de jours",
                value=5,
                precision=0,
            )

            travelers = gr.Number(
                label="Nombre de voyageurs",
                value=2,
                precision=0,
            )

        with gr.Row():
            daily_expenses = gr.Number(
                label="Dépenses quotidiennes par personne",
                value=1200,
            )

            local_transport = gr.Number(
                label="Transport local",
                value=2500,
            )

            budget_activities = gr.Number(
                label="Activités",
                value=4000,
            )

        with gr.Row():
            contingency_rate = gr.Slider(
                minimum=0,
                maximum=0.50,
                value=0.10,
                step=0.05,
                label="Marge de sécurité",
            )

            currency = gr.Textbox(
                label="Devise",
                value="INR",
            )

        budget_button = gr.Button(
            "Estimer le budget",
            variant="primary",
        )

        budget_output = gr.JSON(
            label="Résultat",
        )

        budget_button.click(
            fn=run_budget_estimation,
            inputs=[
                flight_price,
                accommodation_price,
                nights,
                trip_days,
                travelers,
                daily_expenses,
                local_transport,
                budget_activities,
                contingency_rate,
                currency,
            ],
            outputs=budget_output,
        )

    # ---------------------------------------------------------
    # RECOMMANDATION DE DESTINATION
    # ---------------------------------------------------------

    with gr.Tab("🌍 Recommandation"):
        with gr.Row():
            recommendation_budget = gr.Number(
                label="Budget par personne",
                value=25000,
            )

            recommendation_days = gr.Number(
                label="Durée du voyage",
                value=5,
                precision=0,
            )

            recommendation_source_city = gr.Dropdown(
                choices=cities,
                value="Delhi",
                label="Ville de départ",
            )

        with gr.Row():
            preferred_climate = gr.Dropdown(
                choices=climates,
                value="warm",
                label="Climat préféré",
            )

            travel_style = gr.Dropdown(
                choices=travel_styles,
                value="culture",
                label="Style de voyage",
            )

            preferred_activity = gr.Dropdown(
                choices=activities,
                value="food",
                label="Activité préférée",
            )

        recommendation_button = gr.Button(
            "Recommander une destination",
            variant="primary",
        )

        recommendation_output = gr.JSON(
            label="Résultat",
        )

        recommendation_button.click(
            fn=run_recommendation,
            inputs=[
                recommendation_budget,
                recommendation_days,
                preferred_climate,
                travel_style,
                preferred_activity,
                recommendation_source_city,
            ],
            outputs=recommendation_output,
        )


app = gr.mount_gradio_app(
    fastapi_app,
    demo,
    path="/gradio",
)