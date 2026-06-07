"""Tests for src/ingestion/loader.py and schema_detector.py"""
import pytest
import pandas as pd

from src.ingestion.loader import load
from src.ingestion.schema_detector import detect_schema, infer_column_type


class TestLoader:
    def test_load_csv(self, sample_csv):
        df = load(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load("/nonexistent/path/data.csv")

    def test_load_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            load(str(bad))

    def test_load_excel(self, clean_df, tmp_path):
        path = tmp_path / "data.xlsx"
        clean_df.to_excel(path, index=False)
        df = load(str(path))
        assert list(df.columns) == list(clean_df.columns)


class TestSchemaDetector:
    def test_numeric_detection(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert infer_column_type(s) == "numeric"

    def test_boolean_detection(self):
        s = pd.Series([True, False, True])
        assert infer_column_type(s) == "boolean"

    def test_categorical_detection(self):
        # Low cardinality object → categorical
        s = pd.Series(["A", "B", "A", "B", "A"] * 20)
        assert infer_column_type(s) == "categorical"

    def test_text_detection(self):
        # High cardinality object → text
        s = pd.Series([f"unique_value_{i}" for i in range(100)])
        assert infer_column_type(s) == "text"

    def test_detect_schema_returns_all_columns(self, clean_df):
        schema = detect_schema(clean_df)
        assert set(schema.keys()) == set(clean_df.columns)
        assert all(isinstance(v, str) for v in schema.values())
