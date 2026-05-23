import joblib
import pandas as pd

from fastapi import FastAPI
from app.schemas import PenguinInput
from fastapi.responses import RedirectResponse

app = FastAPI()


colonnes_mesures = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]


modele_espece = joblib.load("app/models/species_model.joblib")
modele_anomalie = joblib.load("app/models/anomaly_model.joblib")


@app.get("/")
def home():
    return RedirectResponse(url="/docs")

@app.post("/predict-species")

def predict_species(data: PenguinInput):
    input_df = pd.DataFrame([data.model_dump()], columns=colonnes_mesures)

    prediction = modele_espece.predict(input_df)[0]

    probabilities = modele_espece.predict_proba(input_df)[0]
    confidence = max(probabilities)

    return {
        "predicted_species": prediction,
        "confidence": round(float(confidence), 2)
    }

@app.post("/detect-anomaly")
def detect_anomaly(data: PenguinInput):
    donneesrecues = data.model_dump()

    tableau_pour_modele = pd.DataFrame(
        [donneesrecues],
        columns=colonnes_mesures
    )

    prediction = modele_anomalie.predict(tableau_pour_modele)[0]
    score_anomalie = modele_anomalie.decision_function(tableau_pour_modele)[0]

    if prediction == -1:
        message = "Les mesures semblent inhabituelles."
        est_une_anomalie = True
    else:
        message = "Les mesures semblent normales."
        est_une_anomalie = False

    return {
        "is_anomaly": est_une_anomalie,
        "message": message,
        "anomaly_score": round(float(score_anomalie), 4)
    }

@app.get("/model-info")
def model_info():
    return {
        "classification_model": "RandomForestClassifier",
        "anomaly_model": "IsolationForest",
        "features": colonnes_mesures,
        "classification_accuracy": 0.9565,
        "purpose": "Species prediction and anomaly detection"
    }