import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score



#mesures du modèle : 1)apprendre 2)prédire
colonnes_mesures = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

#colonne que le modèle de classification doit apprendre à prédire
#but prédire l'espèce de pingouin
colonne_cible = "species"


def charger_donnees():
    """
    charge le fichier CSV (pingouins)

    le fichier est lu avec pandas, puis stocké dans un dataframe.
    un df est un tableau de données avec des lignes et des colonnes.

    """

    df = pd.read_csv("app/data/penguins.csv")

    #affiche les premières lignes 
    print(df.head())

    #affiche les colonnes,types de données, et manquants
    print(df.info())

    return df


def nettoyer_donnees(df):
    """
    nettoie les données avant l'entraînement

    car certains pengouins ont des valeurs manquantes
    un modèle de machine learning ne peut pas avoir un apprantissage de qualité avec des données vides

    on supprime seulement les lignes où il manque une donnée dans les colonnes utiles tel que :
    -les colonnes de mesures et la colonne cible species

    """

    colonnes_utiles = colonnes_mesures + [colonne_cible]

    print("nombre de lignes avant le nettoyage :", len(df))

    #dropna(subset=...) sert à supprimer les lignes contenant une valur manquante
    df = df.dropna(subset=colonnes_utiles)

    print("nombre de lignes après le nettoyage :", len(df))

    return df


def preparer_donnees_classification(df):
    """
    prépare les données afin d'netrainer le modèle servant à la classification

    séparation des données:

    X = les données d'entrée mesures du pingouin
    

    y = la réponse attendue soit l'espèce du pengouin
        
    """

    # X colonnes utilisées par le modèle afin de réaliser la prédiction
    X = df[colonnes_mesures]

    # y reponse que le modèle doit apprendre
    y = df[colonne_cible]

    print("Données utilisées pour prédire :")
    print(X.head())

    print("Espèces à prédire :")
    print(y.head())

    return X, y


def separer_donnees(X, y):
    """
    Separation en deux groupes :
    - une part pour entraîner le modèle ;
    - une autre part afin de tester le modèle.

    Pourquoi séparer les données ?

    le but est que le modèle apprennent pas les exemple par coeur, 
    il ne faut pas tester le modèle sur les mêmes données que celles utilisées à l'entrainement. 

    on met alors de côté une partie des données, afin de verifier si le modèle est capable de prédire.
    de manière rigoureuse, sur les données vues à l'entrainement

    X contient les mesures des pingouins.
    y contient les espèces correspondantes aux mesures.

    le modèle apprend avec :
    - X_train : mesures utilisées pour l'entraînement ;
    - y_train : espèces correctes utilisées pour l'entraînement.

    ensuite, on teste avec :
    - X_test : mesures gardées pour le test ;
    - y_test : espèces correctes attendues pour vérifier les prédictions.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,              #les données d'entrée : mesures des pingouins
        y,              #la réponse attendue : espèce du pingouin
        test_size=0.2,  #20 % des données sont mises de côté afin de tester le modèle
        random_state=42 #permet de faire prospérer le même découpage 
    )

    print("nombre de lignes pour entraîner :", len(X_train))
    print("nombre de lignes pour tester :", len(X_test))

    return X_train, X_test, y_train, y_test


def entrainer_modele_espece(X_train, y_train, X_test, y_test):
    """
    entrainer un modèle de classification afin de prédir l'espèce de pengouin

    le modèle utilisé est RandomForestClassifier

    une forêt aléatoire contient plusieurs arbres de décision 
    chaque arbre propose une prédiction, ensuite le modèle chosiit la réponse qui revient le plus

    ici, le modèle apprend à lier des mesures biologiques à une espèce de pengouin :
    - bill_length_mm ;
    - bill_depth_mm ;
    - flipper_length_mm ;
    - body_mass_g

    """

    modele = RandomForestClassifier(
        n_estimators=120, #n nombre d'arbres utilisées dans la forêt
        random_state=42   #resultat devient alors reproductible 
    )

    #le modèle apprend avec les données liées à l'entrainement
    modele.fit(X_train, y_train)

    #le modèle prédit les pescèces venant des données de test
    predictions = modele.predict(X_test)

    #on dresse une compaison entre les prédictions et les vraies réponses y_test.
    #accuracy indique le pourcentage de bonnes prédictions
    precision = accuracy_score(y_test, predictions)

    print("Précision du modèle :", precision)

    #sauvegarde le modèle pour pouvoir le réutiliser au sein de l'api fast api
    joblib.dump(modele, "app/models/species_model.joblib")
    print("Modèle de classification sauvegardé.")

    return modele


def entrainer_modele_anomalie(df):
    """
    entraîne un modèle de détection d'anomalies

    le modèle utilisé est IsolationForest

    ce modèle ci n'a pas besoin de colonnes cibles
    il recherche par lui même les points inabituels

    ex :
    - un pingouin avec une masse extrêmement élevée ;
    - une longueur de nageoire irréaliste ;
    - une combinaison de mesures très éloignée des données normales

    """

    #anomaly detection utilise des mesures dites numériques
    donnees_mesures = df[colonnes_mesures]

    modele = IsolationForest(
        contamination=0.05, #estimation : env 5 % données atypiques
        random_state=42 #pour resultat reproductible
    )

    #le modèle apprend la structure des données 
    modele.fit(donnees_mesures)

    # sauvegarde du modèle
    joblib.dump(modele, "app/models/anomaly_model.joblib")
    print("Modèle de détection d'anomalies sauvegardé.")

    return modele


if __name__ == "__main__":
    """
    ce bloc est exécuté si lancement avec :

    python3 app/train_models.py

    exécuter les étapes

    """

    #étape 1 : charge données depuis CSV.
    df = charger_donnees()

    #étape 2 : supp les lignes incomplètes
    df = nettoyer_donnees(df)

    #étape 3 : sépare les mesures X et la réponse y
    X, y = preparer_donnees_classification(df)

    #étape 4 : sépare les données de l'entrainement et du test
    X_train, X_test, y_train, y_test = separer_donnees(X, y)

    #étape 5 : entraîne puis sauvegarde le modèle de classification
    modele_espece = entrainer_modele_espece(X_train, y_train, X_test, y_test)

    #étape 6 : entraîne et sauvegarde le modèle de détection d'anomalies
    modele_anomalie = entrainer_modele_anomalie(df)