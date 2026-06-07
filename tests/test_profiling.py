"""Tests for profiling dimensions, issue detection, and the pipeline."""
import pytest
import pandas as pd

from src.profiling.dimensions import completeness, uniqueness, validity, consistency
from src.profiling.pipeline import run
from src.models import DatasetProfile


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
        assert uniqueness(s) == pytest.approx(1 / 5)

    def test_validity_numeric_clean(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert validity(s, "numeric") == 1.0

    def test_validity_numeric_with_invalid(self):
        s = pd.Series(["1.0", "2.0", "abc", "3.0"])
        score = validity(s, "numeric")
        assert score == pytest.approx(0.75)

    def test_consistency_no_outliers(self):
        s = pd.Series([10.0, 11.0, 10.5, 10.2, 10.8])
        assert consistency(s, "numeric") == 1.0

    def test_consistency_with_outlier(self):
        normal = [10.0] * 50
        s = pd.Series(normal + [9999.0])
        score = consistency(s, "numeric")
        assert score < 1.0

    def test_consistency_mixed_case(self):
        s = pd.Series(["Leeds", "leeds", "LEEDS", "london", "London"])
        score = consistency(s, "categorical")
        assert score < 1.0


class TestPipeline:
    def test_run_returns_dataset_profile(self, sample_csv):
        profile = run(sample_csv)
        assert isinstance(profile, DatasetProfile)

    def test_profile_has_correct_shape(self, sample_csv, clean_df):
        profile = run(sample_csv)
        assert profile.n_rows == len(clean_df)
        assert profile.n_cols == len(clean_df.columns)

    def test_profile_has_all_columns(self, sample_csv, clean_df):
        profile = run(sample_csv)
        profiled_names = {c.col_name for c in profile.columns}
        assert profiled_names == set(clean_df.columns)

    def test_dimension_scores_in_range(self, sample_csv):
        profile = run(sample_csv)
        for d, score in profile.dimension_scores.items():
            assert 0.0 <= score <= 1.0, f"{d} score out of range: {score}"

    def test_overall_score_in_range(self, sample_csv):
        profile = run(sample_csv)
        assert 0.0 <= profile.overall_score <= 1.0

    def test_summary_text_not_empty(self, sample_csv):
        profile = run(sample_csv)
        assert len(profile.summary_text) > 0

    def test_dirty_data_lower_score(self, tmp_path):
        """A dataset with known issues should score lower than a clean version of same columns."""
        clean = pd.DataFrame({
            "id":    [1, 2, 3, 4, 5],
            "score": ["0.9", "0.8", "0.7", "0.6", "0.5"],
        })
        dirty = pd.DataFrame({
            "id":    [1, 1, None, 4, 5],    # duplicate + missing
            "score": ["0.9", "0.8", "abc", None, "0.5"],  # invalid + missing
        })
        clean_path = tmp_path / "clean.csv"
        dirty_path = tmp_path / "dirty.csv"
        clean.to_csv(clean_path, index=False)
        dirty.to_csv(dirty_path, index=False)

        clean_profile = run(str(clean_path))
        dirty_profile = run(str(dirty_path))
        assert dirty_profile.overall_score < clean_profile.overall_score
