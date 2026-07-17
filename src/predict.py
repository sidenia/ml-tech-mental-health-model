"""Carrega o modelo salvo e faz predições.

Uso programático:
    from src.predict import predict_one
    predict_one({"age": 29, "gender": "male", ...})
"""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/model.joblib")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} não existe. Rode antes: python -m src.train"
        )
    return joblib.load(MODEL_PATH)


def predict_one(features: dict) -> dict:
    """Prediz para um único registro.

    features: dict com as colunas de data/processed/clean.csv (menos o alvo).
    Retorna {"treatment": "yes"|"no", "probability": float}.
    """
    model = load_model()
    X = pd.DataFrame([features])
    proba = float(model.predict_proba(X)[0, 1])
    return {"treatment": "yes" if proba >= 0.5 else "no", "probability": round(proba, 3)}
