from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationReport:
    missing_ratio: dict[str, float]
    duplicate_rows: int
    outlier_counts: dict[str, int]


def validate_frame(df: pd.DataFrame, numeric_cols: list[str]) -> ValidationReport:
    missing = (df.isna().mean()).to_dict()
    duplicates = int(df.duplicated().sum())
    outliers: dict[str, int] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            outliers[col] = 0
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers[col] = int(((series < low) | (series > high)).sum())
    return ValidationReport(missing_ratio=missing, duplicate_rows=duplicates, outlier_counts=outliers)
