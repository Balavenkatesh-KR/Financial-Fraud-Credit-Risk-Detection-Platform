from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_feature_importance(model, X: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        names = X.columns
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
        names = X.columns
    else:
        return []

    ranking = sorted(zip(names, values), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"feature": str(f), "importance": float(v)} for f, v in ranking]


def local_explanation(pipe, row: pd.DataFrame, top_n: int = 3) -> list[dict[str, Any]]:
    # business-friendly fallback explanation when exact SHAP on transformed matrix is expensive.
    model = pipe.named_steps["model"]
    prepped = pipe.named_steps["prep"].transform(row)
    if not hasattr(model, "predict_proba"):
        return []

    try:
        import shap

        explainer = shap.Explainer(model)
        shap_values = explainer(prepped)
        vals = np.abs(np.asarray(shap_values.values)[0])
        names = [f"feature_{i}" for i in range(vals.shape[0])]
        ranking = sorted(zip(names, vals), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"feature": f, "impact": float(v)} for f, v in ranking]
    except Exception:
        score = float(pipe.predict_proba(row)[:, 1][0])
        return [
            {
                "feature": "model_score",
                "impact": round(score, 4),
                "note": "Fallback explanation. Enable full SHAP artifacts offline for compliance packs.",
            }
        ]
