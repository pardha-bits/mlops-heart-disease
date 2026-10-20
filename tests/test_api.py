
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Heart Disease Prediction API"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["label"] in [
        "heart disease present",
        "no heart disease detected",
    ]
    assert set(data["probabilities"].keys()) == {"0", "1"}

    probability_sum = sum(data["probabilities"].values())
    assert abs(probability_sum - 1.0) < 1e-6


def test_prediction_rejects_missing_features():
    response = client.post("/predict", json={"age": 63})
    assert response.status_code == 422
