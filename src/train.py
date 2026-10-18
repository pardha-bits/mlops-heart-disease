from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

if "target" not in df.columns:
    raise ValueError("Dataset must contain a 'target' column.")

X = df.drop(columns=["target"]).apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df["target"], errors="coerce")

valid = y.notna()
X, y = X.loc[valid], y.loc[valid].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numeric_features = X.columns.tolist()

logistic_preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features)
])

forest_preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features)
])

models = {
    "logistic_regression": Pipeline([
        ("preprocessor", logistic_preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]),
    "random_forest": Pipeline([
        ("preprocessor", forest_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )),
    ]),
}

mlflow_db = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
mlflow.set_experiment("heart-disease-classification")

results = []
best_model = None
best_name = None
best_f1 = -1.0

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_param("model_name", name)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_metrics(metrics)

        cm_path = ARTIFACT_DIR / f"{name}_confusion_matrix.png"
        report_path = ARTIFACT_DIR / f"{name}_classification_report.txt"

        ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        plt.title(f"{name}: Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        report_path.write_text(
            classification_report(y_test, predictions, zero_division=0)
        )

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_path))
        mlflow.sklearn.log_model(sk_model=model, name="model", serialization_format="pickle",)

        results.append({"model": name, **metrics})

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_model = model

joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

results_df = pd.DataFrame(results)
results_df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nSelected model by test F1: {best_name}")
print(f"Saved model: {MODEL_DIR / 'best_model.joblib'}")
print(f"Tracking directory: {PROJECT_ROOT / 'mlruns'}")
