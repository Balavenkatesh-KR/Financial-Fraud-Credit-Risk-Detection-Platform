from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FraudRequest(BaseModel):
    customer_id: str
    amount: float = Field(gt=0)
    transaction_hour: int = Field(ge=0, le=23)
    merchant_category: str
    geo_distance_km: float = Field(ge=0)
    txn_count_1h: int = Field(ge=0)
    txn_count_24h: int = Field(ge=0)
    device_change_flag: int = Field(ge=0, le=1)


class CreditRiskRequest(BaseModel):
    customer_id: str
    monthly_income: float = Field(gt=0)
    monthly_emi: float = Field(gt=0)
    credit_limit: float = Field(gt=0)
    credit_used: float = Field(ge=0)
    bureau_score: int = Field(ge=300, le=900)
    dpd_m1: int = Field(ge=0)
    dpd_m2: int = Field(ge=0)
    dpd_m3: int = Field(ge=0)
    loan_tenure_months: int = Field(gt=0)


class PredictionResponse(BaseModel):
    score: float
    threshold: float
    decision: str
    explanation: list[dict[str, Any]]
