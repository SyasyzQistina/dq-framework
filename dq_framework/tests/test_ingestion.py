import pytest
import pandas as pd
from src.ingestion.loader import load
from src.ingestion.schema_detector import infer_column_type, detect_schema


class TestLoader:
    def test_load_csv(self, tmp_path):
        # Create a temp CSV and load it
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        result = load(str(path))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load("/nonexistent/file.csv")

    def test_load_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "file.txt"
        path.write_text("hello")
        with pytest.raises(ValueError):
            load(str(path))


class TestSchemaDetector:
    def test_numeric_detection(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert infer_column_type(s) == "numeric"

    def test_boolean_detection(self):
        s = pd.Series([True, False, True])
        assert infer_column_type(s) == "boolean"

    def test_categorical_detection(self):
        s = pd.Series(["A", "B", "A", "B"] * 20)
        assert infer_column_type(s) == "categorical"

    def test_comma_number_detection(self):
        s = pd.Series([" 99,469 ", " 92,156 ", " 5,897 ", " 713 "])
        assert infer_column_type(s) == "numeric"

    def test_detect_schema_keys(self):
        df = pd.DataFrame({"age": [1, 2], "name": ["a", "b"]})
        schema = detect_schema(df)
        assert set(schema.keys()) == {"age", "name"}