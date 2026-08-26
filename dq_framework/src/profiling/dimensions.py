"""Quality dimension scoring functions.

Each function takes a pandas Series (a single column) and returns a
float score between 0.0 and 1.0, where 1.0 means perfect quality on
that dimension. Used by the profiling pipeline (see pipeline.py) to
build a ColumnProfile for every column in a dataset.
"""

from __future__ import annotations

import pandas as pd


def completeness(series: pd.Series) -> float:
    """Proportion of non-null values in the series.

    1.0 = no missing values, 0.0 = every value is missing.
    """
    if len(series) == 0:
        return 0.0
    return round(series.notna().sum() / len(series), 4)


def uniqueness(series: pd.Series) -> float:
    """Proportion of unique values among non-null entries.

    Intended for non-numeric columns — repeated numeric values (e.g.
    time-series readings) are expected and not a quality issue, so
    this is skipped for numeric columns upstream in the pipeline.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 1.0
    return round(non_null.nunique() / len(non_null), 4)


def validity(series: pd.Series, inferred_type: str) -> float:
    """Proportion of values that conform to the column's inferred type.

    Numeric columns: strips comma thousands-separators and whitespace
    before attempting conversion to float. Minus signs are left as-is
    so negative numbers parse correctly.
    Datetime columns: checked via pandas' own datetime parser.
    All other types: treated as always valid (no format check applies).
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 1.0

    if inferred_type == "numeric":
        cleaned = (
            non_null.astype(str)
            .str.replace(",", "")
            .str.strip()
        )
        valid = pd.to_numeric(cleaned, errors="coerce").notna().sum()
        return round(valid / len(non_null), 4)

    if inferred_type == "datetime":
        valid = pd.to_datetime(non_null, errors="coerce").notna().sum()
        return round(valid / len(non_null), 4)

    return 1.0


def consistency(series: pd.Series, inferred_type: str) -> float:
    """Consistency score — meaning differs by column type.

    Numeric columns: 1.0 minus the proportion of IQR-based outliers,
    using a 3.0 multiplier (wider than the standard 1.5x) to avoid
    over-flagging legitimately wide-ranging datasets. Columns with
    fewer than 4 non-null values are treated as consistent by default,
    since IQR isn't meaningful on that few points.

    Categorical/text columns: 1.0 minus the proportion of values that
    are duplicates once case and surrounding whitespace are normalised
    (e.g. 'Leeds', 'leeds', 'LEEDS' would count as one inconsistency).

    All other types: treated as always consistent.
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
        original = non_null.nunique()
        normalised = non_null.str.strip().str.lower().nunique()
        if original == 0:
            return 1.0
        inconsistent = original - normalised
        return round(1 - inconsistent / original, 4)

    return 1.0