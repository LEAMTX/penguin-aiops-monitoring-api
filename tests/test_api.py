from fastapi.testclient import TestClient

from app.main import app


client_test = TestClient(app)


def test_route_health():
    response = client_test.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    response = client_test.get("/model-info")

    assert response.status_code == 200
    assert response.json()["classification_model"] == "RandomForestClassifier"


def test_predict_species_endpoint():
    mesures_pingouin_predict = {
        "bill_length_mm": 50.0,
        "bill_depth_mm": 16.3,
        "flipper_length_mm": 220.0,
        "body_mass_g": 5200.0
    }

    response = client_test.post("/predict-species", json=mesures_pingouin_predict)

    assert response.status_code == 200
    assert "predicted_species" in response.json()
    assert "confidence" in response.json()


def test_detect_anomaly_endpoint():
    mesures_pingouin_anomaly = {
        "bill_length_mm": 128.0,
        "bill_depth_mm": 55.0,
        "flipper_length_mm": 405.0,
        "body_mass_g": 20000.0
    }

    response = client_test.post("/detect-anomaly", json=mesures_pingouin_anomaly)

    assert response.status_code == 200
    assert "is_anomaly" in response.json()
    assert "anomaly_score" in response.json()