
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score


colonnes_mesures = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

colonne_cible = "species"


def charger_donnees():
    df = pd.read_csv("app/data/penguins.csv")

    print(df.head())
    print(df.info())

    return df


def nettoyer_donnees(df):
    colonnes_utiles = colonnes_mesures + [colonne_cible]

    print("Nombre de lignes avant le nettoyage :", len(df))

    df = df.dropna(subset=colonnes_utiles)

    print("Nombre de lignes après le nettoyage :", len(df))

    return df


def preparer_donnees_classification(df):
    X = df[colonnes_mesures]
    y = df[colonne_cible]

    print("Données utilisées pour prédire :")
    print(X.head())

    print("Espèces à prédire :")
    print(y.head())

    return X, y


def separer_donnees(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Nombre de lignes pour entraîner :", len(X_train))
    print("Nombre de lignes pour tester :", len(X_test))

    return X_train, X_test, y_train, y_test


def entrainer_modele_espece(X_train, y_train, X_test, y_test):
    modele = RandomForestClassifier(
        n_estimators=120,
        random_state=42
    )

    modele.fit(X_train, y_train)

    predictions = modele.predict(X_test)
    precision = accuracy_score(y_test, predictions)

    print("Précision du modèle :", precision)

    joblib.dump(modele, "app/models/species_model.joblib")
    print("Modèle de classification sauvegardé.")

    return modele


def entrainer_modele_anomalie(df):
    donnees_mesures = df[colonnes_mesures]

    modele = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    modele.fit(donnees_mesures)

    joblib.dump(modele, "app/models/anomaly_model.joblib")
    print("Modèle de détection d'anomalies sauvegardé.")

    return modele


if __name__ == "__main__":
    df = charger_donnees()
    df = nettoyer_donnees(df)

    X, y = preparer_donnees_classification(df)
    X_train, X_test, y_train, y_test = separer_donnees(X, y)

    modele_espece = entrainer_modele_espece(X_train, y_train, X_test, y_test)
    modele_anomalie = entrainer_modele_anomalie(df)