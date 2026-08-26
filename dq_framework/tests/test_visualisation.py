import pathlib

import pytest

from src.models import DatasetProfile, ColumnProfile, QualityIssue
from src.visualisation import charts


def _make_profile(columns, dimension_scores=None, issues_by_col=None):
    """Build a minimal DatasetProfile for chart-generation tests."""
    profile = DatasetProfile(
        dataset_name="Test",
        source_path="test.csv",
        n_rows=10,
        n_cols=len(columns),
    )
    profile.columns = columns
    profile.dimension_scores = dimension_scores or {
        "completeness": 0.9, "uniqueness": 0.8,
        "validity": 1.0, "consistency": 0.95,
    }
    if issues_by_col:
        for col, issues in issues_by_col.items():
            for c in profile.columns:
                if c.col_name == col:
                    c.issues = issues
    return profile


NUMERIC_COL = ColumnProfile(
    col_name="Latitude", inferred_type="numeric",
    pct_missing=0.3, completeness=0.997, uniqueness=1.0,
    validity=1.0, consistency=0.94, top_values={},
)
TEXT_COL = ColumnProfile(
    col_name="Postcode", inferred_type="text",
    pct_missing=0.0, completeness=1.0, uniqueness=0.19,
    validity=1.0, consistency=1.0,
    top_values={"LS1 2TW": 149, "LS1 5AA": 130},
)


class TestScoreColor:

    def test_good_threshold(self):
        assert charts._score_color(0.95) == charts.GOOD
        assert charts._score_color(1.0) == charts.GOOD

    def test_warn_threshold(self):
        assert charts._score_color(0.80) == charts.WARN
        assert charts._score_color(0.94) == charts.WARN

    def test_bad_threshold(self):
        assert charts._score_color(0.79) == charts.BAD
        assert charts._score_color(0.0) == charts.BAD


class TestMissingnessHeatmap:

    def test_creates_file(self, tmp_path):
        profile = _make_profile([NUMERIC_COL, TEXT_COL])
        path = charts.missingness_heatmap(profile, str(tmp_path))
        assert pathlib.Path(path).exists()
        assert pathlib.Path(path).suffix == ".png"


class TestDimensionRadar:

    def test_creates_file(self, tmp_path):
        profile = _make_profile([NUMERIC_COL, TEXT_COL])
        path = charts.dimension_radar(profile, str(tmp_path))
        assert pathlib.Path(path).exists()


class TestColumnScoresBar:

    def test_creates_file(self, tmp_path):
        profile = _make_profile([NUMERIC_COL, TEXT_COL])
        path = charts.column_scores_bar(profile, str(tmp_path))
        assert pathlib.Path(path).exists()


class TestIssueSeveritySummary:

    def test_creates_file_with_no_issues(self, tmp_path):
        profile = _make_profile([NUMERIC_COL, TEXT_COL])
        path = charts.issue_severity_summary(profile, str(tmp_path))
        assert pathlib.Path(path).exists()

    def test_creates_file_with_issues(self, tmp_path):
        issue = QualityIssue(
            issue_type="missing_values", severity="high",
            affected_rows=5, pct_affected=50.0,
            description="desc", suggested_action="action",
        )
        col = ColumnProfile(col_name="X", inferred_type="numeric", issues=[issue])
        profile = _make_profile([col])
        path = charts.issue_severity_summary(profile, str(tmp_path))
        assert pathlib.Path(path).exists()


class TestBoxplots:

    def test_returns_none_when_no_numeric_columns(self, tmp_path):
        profile = _make_profile([TEXT_COL])
        result = charts.boxplots(profile, str(tmp_path))
        assert result is None

    def test_creates_file_when_numeric_columns_present(self, tmp_path):
        profile = _make_profile([NUMERIC_COL, TEXT_COL])
        path = charts.boxplots(profile, str(tmp_path))
        assert path is not None
        assert pathlib.Path(path).exists()


class TestFrequencyPlots:

    def test_returns_none_when_no_categorical_columns(self, tmp_path):
        profile = _make_profile([NUMERIC_COL])
        result = charts.frequency_plots(profile, str(tmp_path))
        assert result is None

    def test_returns_none_when_categorical_column_has_no_top_values(self, tmp_path):
        empty_cat = ColumnProfile(
            col_name="Empty", inferred_type="categorical", top_values={},
        )
        profile = _make_profile([empty_cat])
        result = charts.frequency_plots(profile, str(tmp_path))
        assert result is None

    def test_creates_file_when_categorical_columns_present(self, tmp_path):
        profile = _make_profile([NUMERIC_COL, TEXT_COL])
        path = charts.frequency_plots(profile, str(tmp_path))
        assert path is not None
        assert pathlib.Path(path).exists()

    def test_limits_to_two_columns_max(self, tmp_path):
        cat1 = ColumnProfile(col_name="A", inferred_type="text", top_values={"x": 1})
        cat2 = ColumnProfile(col_name="B", inferred_type="text", top_values={"y": 1})
        cat3 = ColumnProfile(col_name="C", inferred_type="text", top_values={"z": 1})
        profile = _make_profile([cat1, cat2, cat3])
        path = charts.frequency_plots(profile, str(tmp_path))
        assert path is not None
        assert pathlib.Path(path).exists()


class TestGenerateAll:

    def test_returns_six_charts_when_all_types_present(self, tmp_path):
        profile = _make_profile([NUMERIC_COL, TEXT_COL])
        paths = charts.generate_all(profile, str(tmp_path))
        assert len(paths) == 6
        for p in paths:
            assert pathlib.Path(p).exists()

    def test_excludes_none_results_when_no_numeric_or_categorical(self, tmp_path):
        # A column that is neither numeric nor categorical/text (e.g.
        # a datetime-only dataset) should still produce the four
        # charts that don't depend on those types, with boxplots and
        # frequency_plots both omitted rather than returning None
        # entries in the list.
        date_col = ColumnProfile(col_name="Date", inferred_type="datetime")
        profile = _make_profile([date_col])
        paths = charts.generate_all(profile, str(tmp_path))
        assert None not in paths
        assert len(paths) == 4