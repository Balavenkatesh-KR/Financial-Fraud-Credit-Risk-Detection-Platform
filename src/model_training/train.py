from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.io import save_joblib
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass
class TrainingArtifacts:
    best_model_name: str
    model_path: str
    auc_scores: dict[str, float]


def _build_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    features = df.drop(columns=[target_col])
    cat_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in features.columns if c not in cat_cols]

    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)]
    )


def train_and_select(
    df: pd.DataFrame,
    target_col: str,
    experiment_name: str,
    artifact_dir: str,
    task_name: str,
) -> TrainingArtifacts:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = _build_preprocessor(df, target_col)

    models: dict[str, Any] = {
        "logreg": LogisticRegression(max_iter=500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced_subsample", random_state=42
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
        )

    mlflow.set_experiment(experiment_name)
    auc_scores: dict[str, float] = {}
    trained: dict[str, Pipeline] = {}

    for name, estimator in models.items():
        with mlflow.start_run(run_name=f"{task_name}_{name}"):
            pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, preds)
            auc_scores[name] = float(auc)
            trained[name] = pipe
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_param("model", name)
            logger.info("%s AUC = %.4f", name, auc)

    best = max(auc_scores, key=auc_scores.get)
    best_pipeline = trained[best]
    model_path = str(Path(artifact_dir) / f"{task_name}_pipeline.joblib")
    save_joblib(best_pipeline, model_path)
    return TrainingArtifacts(best_model_name=best, model_path=model_path, auc_scores=auc_scores)
