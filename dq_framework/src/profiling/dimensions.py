from __future__ import annotations
import pandas as pd


def completeness(series: pd.Series) -> float:
    """Fraction of non-null values. 1.0 = no missing values."""
    if len(series) == 0:
        return 0.0
    return round(series.notna().sum() / len(series), 4)


def uniqueness(series: pd.Series) -> float:
    """Fraction of unique values among non-null entries."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 1.0
    return round(non_null.nunique() / len(non_null), 4)


def validity(series: pd.Series, inferred_type: str) -> float:
    """Fraction of values that match the inferred type."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 1.0

    if inferred_type == "numeric":
        cleaned = non_null.astype(str).str.replace(",", "").str.strip().str.replace("-", "0")
        valid = pd.to_numeric(cleaned, errors="coerce").notna().sum()
        return round(valid / len(non_null), 4)

    if inferred_type == "datetime":
        valid = pd.to_datetime(non_null, errors="coerce").notna().sum()
        return round(valid / len(non_null), 4)

    return 1.0


def consistency(series: pd.Series, inferred_type: str) -> float:
    """Check for outliers (numeric) or mixed casing (categorical)."""
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
        original = non_null.nunique()
        normalised = non_null.str.strip().str.lower().nunique()
        if original == 0:
            return 1.0
        inconsistent = original - normalised
        return round(1 - inconsistent / original, 4)

    return 1.0