from __future__ import annotations
import pandas as pd


def infer_column_type(series: pd.Series) -> str:
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"

    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    # For text/object columns
    sample = series.dropna().head(100).astype(str)

    # Check if it looks like dates
    date_like = sample.str.match(r"^\d{4}[-/]\d{2}[-/]\d{2}").mean()
    if date_like > 0.8:
        return "datetime"

    # Low cardinality = categorical
    n_unique = series.nunique()
    n_total = series.count()
    # Check for comma-formatted numbers like "103,812"
    cleaned = sample.str.replace(",", "").str.strip().str.replace("-", "0")
    numeric_like = pd.to_numeric(cleaned, errors="coerce").notna().mean()
    if numeric_like > 0.8:
        return "numeric"

    if n_total > 0 and (n_unique / n_total) < 0.1:
        return "categorical"
    return "text"


def detect_schema(df: pd.DataFrame) -> dict[str, str]:
    return {col: infer_column_type(df[col]) for col in df.columns}