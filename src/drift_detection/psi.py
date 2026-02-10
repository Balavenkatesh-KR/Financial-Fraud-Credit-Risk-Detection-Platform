from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(expected.quantile(quantiles).values)
    if len(cut_points) < 3:
        return 0.0

    exp_bins = pd.cut(expected, bins=cut_points, include_lowest=True)
    act_bins = pd.cut(actual, bins=cut_points, include_lowest=True)

    exp_pct = exp_bins.value_counts(normalize=True).sort_index()
    act_pct = act_bins.value_counts(normalize=True).sort_index()

    eps = 1e-6
    psi_vals = (act_pct + eps - exp_pct + eps) * np.log((act_pct + eps) / (exp_pct + eps))
    return float(psi_vals.sum())
