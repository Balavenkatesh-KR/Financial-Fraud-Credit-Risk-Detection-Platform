import pandas as pd

from src.feature_engineering.features import add_credit_features, add_geo_anomaly_feature


def test_add_geo_anomaly_feature():
    df = pd.DataFrame(
        {
            "home_lat": [12.9716],
            "home_lon": [77.5946],
            "txn_lat": [19.0760],
            "txn_lon": [72.8777],
        }
    )
    out = add_geo_anomaly_feature(df)
    assert "geo_distance_km" in out.columns
    assert out["geo_distance_km"].iloc[0] > 0


def test_add_credit_features():
    df = pd.DataFrame(
        {
            "monthly_income": [50000],
            "monthly_emi": [10000],
            "credit_used": [20000],
            "credit_limit": [100000],
            "dpd_m1": [0],
            "dpd_m2": [5],
            "dpd_m3": [10],
            "merchant_category": ["retail"],
            "label_proxy": [1],
        }
    )
    out = add_credit_features(df)
    assert out["income_to_emi_ratio"].iloc[0] == 5
    assert out["utilization_ratio"].iloc[0] == 0.2
