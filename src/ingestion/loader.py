"""
Ingestion layer: load any supported file format into a normalised pandas DataFrame.
Supported: CSV, Excel (.xlsx/.xls), JSON, Parquet, direct URL (CSV/JSON).
"""
from __future__ import annotations
import pathlib
import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


def load(source: str) -> pd.DataFrame:
    """
    Load a dataset from a file path or URL into a DataFrame.

    Args:
        source: local file path or HTTP/HTTPS URL.

    Returns:
        Raw pandas DataFrame.

    Raises:
        ValueError: if the format is not supported.
        FileNotFoundError: if a local path does not exist.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return _load_url(source)

    path = pathlib.Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    loaders = {
        ".csv":     lambda p: pd.read_csv(p),
        ".xlsx":    lambda p: pd.read_excel(p, engine="openpyxl"),
        ".xls":     lambda p: pd.read_excel(p, engine="xlrd"),
        ".json":    lambda p: pd.read_json(p),
        ".parquet": lambda p: pd.read_parquet(p),
    }
    return loaders[suffix](path)


def _load_url(url: str) -> pd.DataFrame:
    """Load a CSV or JSON resource from a URL."""
    if url.endswith(".json"):
        return pd.read_json(url)
    # Default: assume CSV (most open data portals serve CSV)
    return pd.read_csv(url)
