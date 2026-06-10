from __future__ import annotations
import pathlib
import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json"}

ENCODINGS = ["utf-8", "latin-1", "windows-1252", "iso-8859-1"]


def _read_csv_safe(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    """Try multiple encodings until one works."""
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode file — unsupported encoding.")


def _detect_header_row(path: pathlib.Path) -> int:
    """
    Scan the first 5 rows to find which one is most likely the header.
    The header row usually has the most non-null string values.
    """
    sample = _read_csv_safe(path, header=None, nrows=5)
    best_row = 0
    best_score = -1

    for i in range(len(sample)):
        row = sample.iloc[i]
        score = 0
        for val in row:
            if pd.isna(val):
                continue
            try:
                float(str(val).replace(",", "").strip())
            except ValueError:
                score += 1
        if score > best_score:
            best_score = score
            best_row = i

    return best_row


def load(source: str) -> pd.DataFrame:
    """
    Load a dataset from a file path into a DataFrame.
    Automatically detects the correct header row and encoding.
    """
    path = pathlib.Path(source)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'")

    if suffix == ".csv":
        header_row = _detect_header_row(path)
        df = _read_csv_safe(path, header=header_row)
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
        df = df.dropna(how="all").reset_index(drop=True)
        return df

    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)

    elif suffix == ".json":
        return pd.read_json(path)