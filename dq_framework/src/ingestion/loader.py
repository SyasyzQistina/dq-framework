from __future__ import annotations
import pathlib
import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json"}


def _detect_header_row(path: pathlib.Path) -> int:
    """
    Scan the first 5 rows to find which one is most likely the header.
    The header row usually has the most non-null string values.
    """
    sample = pd.read_csv(path, header=None, nrows=5)
    best_row = 0
    best_score = -1

    for i in range(len(sample)):
        row = sample.iloc[i]
        # Count values that look like column names (strings, not numbers)
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
    Automatically detects the correct header row.
    """
    path = pathlib.Path(source)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'")

    if suffix == ".csv":
        header_row = _detect_header_row(path)
        df = pd.read_csv(path, header=header_row)
        # Drop columns that are still unnamed after header detection
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
        # Drop rows that are all NaN
        df = df.dropna(how="all").reset_index(drop=True)
        return df

    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)

    elif suffix == ".json":
        return pd.read_json(path)