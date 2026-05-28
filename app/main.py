# joblib charge les modèles sauvegardés en .joblib
import joblib

#pandas crée un DataFrame avec les données reçues
import pandas as pd

#créer l'API
from fastapi import FastAPI

#rediriger la page d'accueil vers /docs
from fastapi.responses import RedirectResponse

#schéma Pydantic qui vérifie les données envoyées par l'utilisateur
from app.schemas import PenguinInputByUser

app = FastAPI()


#liste des colonnes utilisées par les modèles
# mêmes nom que à l'entraînement
colonnes_mesures = [
    "bill_length_mm",        #longueur du bec en millimètres
    "bill_depth_mm",         #profondeur du bec en millimètres
    "flipper_length_mm",     #longueur de la nageoire en millimètres
    "body_mass_g"            #masse corporelle en grammes
]


#charger le modèle entrainé pour prédire l'espèce
modele_espece = joblib.load("app/models/species_model.joblib")

#charger le modèle entraîné pour détecter les anomalies
modele_anomalie = joblib.load("app/models/anomaly_model.joblib")



@app.get("/")
def home():
    # redirection vers la documentation automatique
    return RedirectResponse(url="/docs")



@app.post("/predict-species")
def predict_species(data: PenguinInputByUser):
    #transforme les données reçues en dictionnaire Python
    donnees_pingouin = data.model_dump()

    #transforme le dictionnaire en dataframe pandas
    #modèle scikit-learn attend un tableau avec les mêmes colonnes qu'à l'entraînement
    tableau_mesures = pd.DataFrame(
        [donnees_pingouin],
        columns=colonnes_mesures
    )

    #modèle prédit l'espèce du pingouin
    #[0] permet de récupérer le premier élément du tableau, soit la première prédiction
    espece_predite = modele_espece.predict(tableau_mesures)[0]

    #modèle donne les probabilités pour chaque espèce possible
    probabilites = modele_espece.predict_proba(tableau_mesures)[0]

    #probabilité max = score de confiance
    confiance = max(probabilites)

    #resultat renvoyé frmat json
    return {
        "predicted_species": espece_predite,
        "confidence": round(float(confiance), 2)
    }


# detecte les mesures normales ou inhabituelles
@app.post("/detect-anomaly")
def detect_anomaly(data: PenguinInputByUser):
    #données reçcues--> dictionnaire python
    donnees_pingouin = data.model_dump()

    #données reccues-->dataframe
    #même format qu'à l'entrainement
    tableau_mesures = pd.DataFrame(
        [donnees_pingouin],
        columns=colonnes_mesures
    )

    # IsolationForest renvoit : 1 -> donnée normale et -1 --> une anomalie
    prediction_anomalie = modele_anomalie.predict(tableau_mesures)[0]

    
    # decision function donne score, plus score est bas, plus la données est inhabituelle 
    score_anomalie = modele_anomalie.decision_function(tableau_mesures)[0]

    # true si modèle trouve une anomalie, et renvoit -1
    le_modele_a_trouve_une_anomalie = prediction_anomalie == -1

    #si le modèle a trouvé une anomalie
    if le_modele_a_trouve_une_anomalie:
        est_une_anomalie = True

        message = "Les mesures semblent inhabituelles."

    #sinon normal
    else:
        est_une_anomalie = False

        message = "Les mesures semblent normales."

    #renvoit le resultat en json
    return {
        "is_anomaly": est_une_anomalie,
        "message": message,
        "anomaly_score": round(float(score_anomalie), 4)
    }


@app.get("/model-info")
def model_info():
    #renvoit les informations du projet
    return {
        "classification_model": "RandomForestClassifier",
        "anomaly_model": "IsolationForest",
        "features": colonnes_mesures,
        "classification_accuracy": 0.9565,
        "purpose": "Species prediction and anomaly detection"
    }

@app.get("/health")
def health_check_api():
    return {
        "status":"ok",
        "message" : " Api is running"
    }

@app.get("/metrics")
def show_metrics():
    return {
        "models_loaded": True,
        "classification_model": "RandomForestClassifier",
        "anomaly_model": "IsolationForest",
        "features_count": len(colonnes_mesures),
        "available_features": colonnes_mesures
    }