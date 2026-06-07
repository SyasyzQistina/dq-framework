"""
Schema detection: infer a human-readable type for each column
beyond pandas' dtype (e.g. distinguish 'datetime' stored as str).
"""
from __future__ import annotations
import pandas as pd


def infer_column_type(series: pd.Series) -> str:
    """
    Return one of: "numeric" | "categorical" | "datetime" | "boolean" | "text" | "unknown"
    """
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"

    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
        sample = series.dropna().head(100)

        # Only attempt datetime parse if values look date-like
        str_sample = sample.astype(str)
        date_like = str_sample.str.match(
            r"^\d{4}[-/]\d{2}[-/]\d{2}"
        ).mean()
        if date_like > 0.8:
            try:
                parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
                if parsed.notna().mean() > 0.8:
                    return "datetime"
            except Exception:
                pass

        n_unique = series.nunique()
        n_total = series.count()
        # Heuristic: cardinality < 10% of rows → categorical
        if n_total > 0 and (n_unique / n_total) < 0.1:
            return "categorical"
        return "text"

    return "unknown"


def detect_schema(df: pd.DataFrame) -> dict[str, str]:
    """Return {column_name: inferred_type} for all columns."""
    return {col: infer_column_type(df[col]) for col in df.columns}
