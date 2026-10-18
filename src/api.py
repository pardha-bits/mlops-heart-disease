
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"

model = joblib.load(MODEL_PATH)
FEATURES = pd.read_csv(DATA_PATH).drop(columns=["target"]).columns.tolist()

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps assignment prediction service",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalch: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probabilities: dict[str, float]


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "features": FEATURES,
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    data = request.model_dump()
    row = pd.DataFrame([data], columns=FEATURES)

    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    return {
        "prediction": prediction,
        "label": (
            "heart disease present"
            if prediction == 1
            else "no heart disease detected"
        ),
        "probabilities": {
            str(int(cls)): float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        },
    }
