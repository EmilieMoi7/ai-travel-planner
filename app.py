from fastapi import FastAPI

app = FastAPI(
    title="AI Travel Planner API",
    description="API de prédiction pour les réservations d'hôtels et les prix des vols.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API AI Travel Planner"
    }