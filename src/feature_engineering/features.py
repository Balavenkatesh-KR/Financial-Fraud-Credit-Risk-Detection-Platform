from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def add_transaction_velocity_features(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    ts_col: str = "transaction_ts",
    amount_col: str = "amount",
) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    out = out.sort_values([customer_col, ts_col])
    grouped = out.set_index(ts_col).groupby(customer_col)[amount_col]
    out["txn_count_1h"] = grouped.rolling("1h").count().reset_index(level=0, drop=True).values
    out["txn_count_24h"] = grouped.rolling("24h").count().reset_index(level=0, drop=True).values
    out["txn_amount_7d"] = grouped.rolling("7d").sum().reset_index(level=0, drop=True).fillna(0).values
    return out


def add_geo_anomaly_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["geo_distance_km"] = haversine_km(
        out["home_lat"], out["home_lon"], out["txn_lat"], out["txn_lon"]
    )
    return out


def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["income_to_emi_ratio"] = out["monthly_income"] / np.clip(out["monthly_emi"], 1, None)
    out["utilization_ratio"] = out["credit_used"] / np.clip(out["credit_limit"], 1, None)
    out["delinquency_trend_90d"] = out[["dpd_m1", "dpd_m2", "dpd_m3"]].mean(axis=1)
    out["merchant_risk_score"] = out.groupby("merchant_category")["label_proxy"].transform("mean")
    return out
