from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import ObesityFeatureEngineer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Obesity.csv"
MODEL_PATH = ROOT / "models" / "obesity_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
CLASSIFICATION_REPORT_PATH = ROOT / "reports" / "classification_report.csv"
CONFUSION_MATRIX_PATH = ROOT / "reports" / "confusion_matrix.csv"

NUMERIC_FEATURES = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
CATEGORICAL_FEATURES = ["Gender", "family_history", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "Age_Group", "FCVC_round", "NCP_round", "CH2O_round", "FAF_round", "TUE_round"]
TARGET = "Obesity"


def build_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    return Pipeline([
        ("features", ObesityFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def train():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    candidates = {
        "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []
    fitted_models = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results.append({
            "model": name,
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_f1_macro": float(f1_score(y_test, predictions, average="macro"))
        })
        fitted_models[name] = pipeline
    best = max(results, key=lambda item: item["test_accuracy"])
    best_model = fitted_models[best["model"]]
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions, labels=best_model.classes_)
    pd.DataFrame(report).transpose().to_csv(CLASSIFICATION_REPORT_PATH)
    pd.DataFrame(matrix, index=best_model.classes_, columns=best_model.classes_).to_csv(CONFUSION_MATRIX_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump({
            "best_model": best["model"],
            "classes": best_model.classes_.tolist(),
            "results": results,
            "test_sample_size": int(len(y_test)),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_f1_macro": float(f1_score(y_test, predictions, average="macro")),
            "max_probability_mean": float(probabilities.max(axis=1).mean())
        }, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    train()
