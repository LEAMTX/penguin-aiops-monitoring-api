# Penguin AIOps Monitoring API

## Project goal

This project is a small machine learning project applied to a monitoring use case.

The goal is to create an API able to:

1. predict a penguin species from biological measurements;
2. detect whether the submitted measurements look normal or unusual.

The project is inspired by AIOps workflows: using data, training models, detecting anomalies and exposing results through an API.

## Link with AIOps

Although the dataset is biological, the technical logic is close to infrastructure monitoring.

Penguin measurements can be compared to infrastructure metrics such as:

- CPU usage;
- memory usage;
- latency;
- response time;
- error rate.

The anomaly detection model identifies unusual measurements, similar to how an AIOps system could detect abnormal infrastructure or application behavior.

## Dataset

This project uses the Palmer Penguins dataset.

The dataset contains:

- penguin species;
- bill length;
- bill depth;
- flipper length;
- body mass;
- island;
- sex.

In this project, only the following numerical features are used:

```python
colonnes_mesures = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]
```

The target column for classification is:

```python
colonne_cible = "species"
```

## Technical skills demonstrated

This project demonstrates:

- Python;
- data processing with pandas;
- data cleaning;
- supervised machine learning;
- anomaly detection;
- simple model validation;
- model persistence with joblib;
- API development with FastAPI;
- input validation with Pydantic;
- interactive API documentation with Swagger UI.

## Learning approach

This project was built as a learning project to explore the main steps of an AIOps pipeline.

I focused on:

- understanding the dataset;
- cleaning missing values;
- training a first supervised model;
- validating the model with accuracy;
- training an anomaly detection model;
- exposing both models through FastAPI endpoints.

The goal was not to build a perfect production system, but to demonstrate a concrete understanding of the full AI/ML pipeline: data, model, validation and API serving.

## Project structure

```text
penguin-iaops/
├── app/
│   ├── data/
│   │   └── penguins.csv
│   ├── models/
│   │   ├── species_model.joblib
│   │   └── anomaly_model.joblib
│   ├── main.py
│   ├── schemas.py
│   └── train_models.py
├── doc/
│   └── images/
│       ├── speciespredictiontest.png
│       ├── speciespredictionresult.png
│       ├── anomalydetectiontest.png
│       └── anomalydetectionresult.png
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── Makefile
├── Learningnotes.md
├── README_EN.md
└── README.md
```

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the models

```bash
python3 app/train_models.py
```

This command:

1. loads the dataset;
2. cleans missing values;
3. prepares the classification data;
4. splits the data into train and test sets;
5. trains a RandomForestClassifier model;
6. computes the accuracy;
7. trains an IsolationForest model;
8. saves both models into `app/models`.

## Model result

The classification model reaches an accuracy of approximately:

```text
0.9565
```

This means that around 95.65% of the predictions are correct on the test data.

## Run the API

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Available endpoints

### GET /

Checks that the API is running.

### POST /predict-species

Predicts the penguin species from the submitted measurements.

Input example:

```json
{
  "bill_length_mm": 50.0,
  "bill_depth_mm": 16.3,
  "flipper_length_mm": 220.0,
  "body_mass_g": 5200.0
}
```

Response example:

```json
{
  "predicted_species": "Gentoo",
  "confidence": 1
}
```

### POST /detect-anomaly

Detects whether the submitted measurements look normal or unusual.

Unusual input example:

```json
{
  "bill_length_mm": 120.0,
  "bill_depth_mm": 50.0,
  "flipper_length_mm": 400.0,
  "body_mass_g": 20000.0
}
```

Response example:

```json
{
  "is_anomaly": true,
  "message": "Les mesures semblent inhabituelles.",
  "anomaly_score": -0.1676
}
```

### GET /model-info

Returns information about the models used by the API.

## API test screenshots

### Species prediction test

![Species prediction test](doc/images/speciespredictiontest.png)

### Species prediction result

![Species prediction result](doc/images/speciespredictionresult.png)

### Anomaly detection test

![Anomaly detection test](doc/images/anomalydetectiontest.png)

### Anomaly detection result

![Anomaly detection result](doc/images/anomalydetectionresult.png)

## Result interpretation

The `/predict-species` endpoint returned `Gentoo`, meaning that the model identified the input measurements as close to the Gentoo species.

The `/detect-anomaly` endpoint returned `is_anomaly: true` when intentionally unrealistic measurements were submitted. The negative anomaly score confirms that the model considered the input unusual.

## Project limits

This project is a learning-oriented MVP.

Current limits:

- the dataset is small;
- there is no dashboard yet;
- there are no automated tests yet;
- the model is not deployed in production;
- anomaly detection is based on an estimated 5% contamination rate.

## Possible improvements

- add a Streamlit dashboard;
- add unit tests;
- add a CI/CD pipeline;
- add a monitoring endpoint;
- compare several models;
- add a more detailed classification report.

## Docker

Build the Docker image:

```bash
docker build -t penguin-aiops-api .
```

Run the container:

```bash
docker run -p 8000:8000 penguin-aiops-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Why I built this project

I built this project after reading the AI Ops internship requirements.

The internship mentions anomaly detection, predictive analytics, infrastructure monitoring, model validation, visualization and API endpoints. I wanted to create a small but complete project that demonstrates these technical blocks in a realistic and understandable way.

I chose the Palmer Penguins dataset because it is clean, scientific and easy to explain. Instead of using infrastructure metrics directly, I used biological measurements as simplified monitoring metrics.

The goal was to show that I understand the logic behind an AIOps pipeline:

- collect data;
- clean and prepare data;
- train a model;
- validate the model;
- detect anomalies;
- expose the model through an API;
- test the API through documentation.

This project is not presented as a production-ready AIOps system. It is a learning-oriented MVP designed to demonstrate motivation, technical curiosity and concrete progress.

## Sources

- Palmer Penguins dataset: https://allisonhorst.github.io/palmerpenguins/
- pandas documentation: https://pandas.pydata.org/docs/
- scikit-learn documentation: https://scikit-learn.org/stable/
- FastAPI documentation: https://fastapi.tiangolo.com/
- Pydantic documentation: https://docs.pydantic.dev/
- joblib documentation: https://joblib.readthedocs.io/