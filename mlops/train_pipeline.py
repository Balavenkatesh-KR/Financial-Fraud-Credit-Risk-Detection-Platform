from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.feature_engineering.features import add_credit_features, add_geo_anomaly_feature, add_transaction_velocity_features
from src.model_training.train import train_and_select
from src.utils.config import load_yaml


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def run(task: str, data_path: str, config_path: str = "configs/training.yaml") -> None:
    cfg = load_yaml(config_path)
    artifact_dir = cfg["artifacts"]["model_dir"]
    exp_name = cfg["mlflow"]["experiment_name"]

    df = load_training_data(data_path)
    if task == "fraud":
        if {"transaction_ts", "customer_id", "amount"}.issubset(df.columns):
            df = add_transaction_velocity_features(df)
        if {"home_lat", "home_lon", "txn_lat", "txn_lon"}.issubset(df.columns):
            df = add_geo_anomaly_feature(df)
        target_col = cfg["label_columns"]["fraud"]
    else:
        df = add_credit_features(df)
        target_col = cfg["label_columns"]["credit"]

    artifacts = train_and_select(df, target_col, exp_name, artifact_dir, task)
    print(f"Best model: {artifacts.best_model_name}")
    print(f"Model path: {artifacts.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fraud", "credit"], required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    run(args.task, args.data)
