from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import CreditRiskRequest, FraudRequest, PredictionResponse
from src.explainability.shap_explainer import local_explanation
from src.utils.config import load_yaml
from src.utils.io import load_joblib

cfg = load_yaml("configs/api.yaml")
app = FastAPI(title=cfg["service"]["name"], version=cfg["service"]["version"])


class ModelBundle:
    def __init__(self, path: str, threshold: float = 0.5):
        self.path = Path(path)
        self.threshold = threshold
        self.pipe = load_joblib(path) if self.path.exists() else None

    def predict(self, payload: dict) -> dict:
        if self.pipe is None:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {self.path}")
        row = pd.DataFrame([payload])
        score = float(self.pipe.predict_proba(row)[:, 1][0])
        decision = "high_risk" if score >= self.threshold else "low_risk"
        explanation = local_explanation(self.pipe, row, top_n=cfg["explainability"]["top_features"])
        return {
            "score": round(score, 6),
            "threshold": self.threshold,
            "decision": decision,
            "explanation": explanation,
        }


fraud_model = ModelBundle(cfg["model_paths"]["fraud"], threshold=0.6)
credit_model = ModelBundle(cfg["model_paths"]["credit"], threshold=0.55)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": bool(fraud_model.pipe and credit_model.pipe)}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "fraud_model_path": str(fraud_model.path),
        "credit_model_path": str(credit_model.path),
        "fraud_threshold": fraud_model.threshold,
        "credit_threshold": credit_model.threshold,
    }


@app.post("/predict/fraud", response_model=PredictionResponse)
def predict_fraud(request: FraudRequest):
    return fraud_model.predict(request.model_dump())


@app.post("/predict/credit-risk", response_model=PredictionResponse)
def predict_credit(request: CreditRiskRequest):
    payload = request.model_dump()
    payload["income_to_emi_ratio"] = payload["monthly_income"] / max(payload["monthly_emi"], 1)
    payload["utilization_ratio"] = payload["credit_used"] / max(payload["credit_limit"], 1)
    payload["delinquency_trend_90d"] = (payload["dpd_m1"] + payload["dpd_m2"] + payload["dpd_m3"]) / 3
    payload["merchant_category"] = "not_applicable"
    payload["label_proxy"] = 0
    return credit_model.predict(payload)
