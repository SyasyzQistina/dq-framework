import pytest
import pandas as pd
from src.profiling.dimensions import completeness, uniqueness, validity, consistency
from src.profiling.pipeline import run


class TestDimensions:
    def test_completeness_full(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert completeness(s) == 1.0

    def test_completeness_partial(self):
        s = pd.Series([1, None, 3, None, 5])
        assert completeness(s) == pytest.approx(0.6)

    def test_completeness_empty(self):
        assert completeness(pd.Series([], dtype=float)) == 0.0

    def test_uniqueness_all_unique(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert uniqueness(s) == 1.0

    def test_uniqueness_all_same(self):
        s = pd.Series([1, 1, 1, 1, 1])
        assert uniqueness(s) == pytest.approx(0.2)

    def test_validity_numeric_clean(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert validity(s, "numeric") == 1.0

    def test_validity_numeric_with_commas(self):
        s = pd.Series([" 99,469 ", " 92,156 ", " 5,897 "])
        assert validity(s, "numeric") == 1.0

    def test_consistency_no_outliers(self):
        s = pd.Series([10.0, 11.0, 10.5, 10.2, 10.8])
        assert consistency(s, "numeric") == 1.0

    def test_consistency_mixed_case(self):
        s = pd.Series(["Leeds", "leeds", "LEEDS", "london", "London"])
        assert consistency(s, "categorical") < 1.0


class TestPipeline:
    def test_run_returns_profile(self, tmp_path):
        df = pd.DataFrame({
            "id":   [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "age":  [25, 30, 22, 45, 33],
        })
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        profile = run(str(path))
        assert profile.n_rows == 5
        assert profile.n_cols == 3

    def test_overall_score_in_range(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        profile = run(str(path))
        assert 0.0 <= profile.overall_score <= 1.0

    def test_dirty_scores_lower(self, tmp_path):
        clean = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        dirty = pd.DataFrame({"val": [1, 1, None, None, 5]})
        clean_path = tmp_path / "clean.csv"
        dirty_path = tmp_path / "dirty.csv"
        clean.to_csv(clean_path, index=False)
        dirty.to_csv(dirty_path, index=False)
        assert run(str(dirty_path)).overall_score < run(str(clean_path)).overall_score