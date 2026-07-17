"""API de predição.

Uso:
    uvicorn api.main:app --reload

Docs interativas: http://127.0.0.1:8000/docs
Cada predição é logada em monitoring/predictions.log (JSON lines).
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_one

LOG_PATH = Path("monitoring/predictions.log")

app = FastAPI(
    title="Mental Health Treatment Predictor",
    description="Prevê se um profissional de tech buscará tratamento de saúde mental. Projeto educacional — não é ferramenta de diagnóstico.",
    version="0.1.0",
)


class Features(BaseModel):
    age: int = Field(ge=18, le=75, examples=[29])
    gender: str = Field(examples=["male"])
    self_employed: str = Field(default="No", examples=["No"])
    family_history: str = Field(examples=["Yes"])
    work_interfere: str = Field(default="Unknown", examples=["Sometimes"])
    no_employees: str = Field(examples=["26-100"])
    remote_work: str = Field(examples=["No"])
    tech_company: str = Field(examples=["Yes"])
    benefits: str = Field(examples=["Yes"])
    care_options: str = Field(examples=["No"])


class Prediction(BaseModel):
    treatment: str
    probability: float


def log_prediction(features: dict, result: dict, latency_ms: float) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": features,
        "output": result,
        "latency_ms": round(latency_ms, 1),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: Features) -> dict:
    start = time.perf_counter()
    result = predict_one(features.model_dump())
    log_prediction(features.model_dump(), result, (time.perf_counter() - start) * 1000)
    return result
