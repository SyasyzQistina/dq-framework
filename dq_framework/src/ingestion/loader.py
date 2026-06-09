from __future__ import annotations
import pathlib
import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json"}

def load(source: str) -> pd.DataFrame:
    path = pathlib.Path(source)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'")

    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif suffix == ".json":
        return pd.read_json(path)