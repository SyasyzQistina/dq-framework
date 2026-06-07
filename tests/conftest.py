"""
Shared pytest fixtures for the DQ Framework test suite.
"""
import pandas as pd
import pytest

from src.models import DatasetProfile, ColumnProfile, QualityIssue


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small, clean DataFrame with no quality issues."""
    return pd.DataFrame({
        "id":     [1, 2, 3, 4, 5],
        "name":   ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "age":    [25, 30, 22, 45, 33],
        "city":   ["Leeds", "London", "Leeds", "Bristol", "London"],
        "joined": ["2023-01-01", "2023-02-15", "2023-03-10", "2022-12-01", "2023-04-05"],
    })


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """A DataFrame with deliberate quality problems for testing detectors."""
    return pd.DataFrame({
        "id":     [1, 2, 2, 4, None],           # duplicate + missing
        "name":   ["Alice", "bob", "BOB", None, "Eve"],  # mixed case + missing
        "age":    [25, -999, 22, 45, 33],        # outlier
        "score":  ["0.9", "0.8", "abc", "0.7", "0.6"],  # invalid format
        "date":   ["2023-01-01", "not-a-date", "2023-03-10", None, "2023-04-05"],
    })


@pytest.fixture
def sample_csv(tmp_path, clean_df) -> str:
    """Write clean_df to a temp CSV and return its path."""
    path = tmp_path / "sample.csv"
    clean_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_profile(clean_df) -> DatasetProfile:
    """A minimal DatasetProfile built from clean_df."""
    from src.profiling.pipeline import run as pipeline_run
    import pathlib
    # Write to temp file first
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        clean_df.to_csv(f.name, index=False)
        path = f.name
    profile = pipeline_run(path)
    os.unlink(path)
    return profile
