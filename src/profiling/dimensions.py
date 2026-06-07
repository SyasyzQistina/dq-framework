"""
Quality dimension scores for a single column.
Each function returns a float in [0.0, 1.0].
"""
from __future__ import annotations
import pandas as pd
import re


def completeness(series: pd.Series) -> float:
    """Fraction of non-null values."""
    if len(series) == 0:
        return 0.0
    return round(series.notna().sum() / len(series), 4)


def uniqueness(series: pd.Series) -> float:
    """
    Fraction of non-duplicate values among non-null entries.
    Score of 1.0 means every value is unique.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 1.0
    return round(non_null.nunique() / len(non_null), 4)


def validity(series: pd.Series, inferred_type: str) -> float:
    """
    Fraction of values that are consistent with the inferred type.
    For numeric: values that can be cast to float.
    For datetime: values that can be parsed as dates.
    For categorical/text/boolean: 1.0 (no strict format to violate).
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 1.0

    if inferred_type == "numeric":
        valid = pd.to_numeric(non_null, errors="coerce").notna().sum()
        return round(valid / len(non_null), 4)

    if inferred_type == "datetime":
        try:
            valid = pd.to_datetime(non_null, infer_datetime_format=True, errors="coerce").notna().sum()
            return round(valid / len(non_null), 4)
        except Exception:
            return 0.0

    return 1.0


def consistency(series: pd.Series, inferred_type: str) -> float:
    """
    Heuristic consistency check:
    - Numeric: penalise statistical outliers (IQR method).
    - Categorical: penalise mixed-case variants of same value.
    - Others: 1.0
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 1.0

    if inferred_type == "numeric":
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric) < 4:
            return 1.0
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        outliers = ((numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)).sum()
        return round(1 - outliers / len(numeric), 4)

    if inferred_type in ("categorical", "text"):
        # Check if stripping & lower-casing reduces unique count (mixed case duplicates)
        original_unique = non_null.nunique()
        normalised_unique = non_null.str.strip().str.lower().nunique()
        if original_unique == 0:
            return 1.0
        inconsistent = original_unique - normalised_unique
        return round(1 - inconsistent / original_unique, 4)

    return 1.0
