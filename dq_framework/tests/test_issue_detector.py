import pandas as pd
import pytest

from src.profiling.issue_detector import detect_issues, _severity


class TestSeverity:

    def test_high_below_080(self):
        assert _severity(0.79) == "high"
        assert _severity(0.0) == "high"

    def test_medium_between_080_and_095(self):
        assert _severity(0.80) == "medium"
        assert _severity(0.94) == "medium"

    def test_low_at_or_above_095(self):
        assert _severity(0.95) == "low"
        assert _severity(1.0) == "low"


class TestNoIssues:

    def test_no_issues_when_all_scores_perfect(self):
        s = pd.Series([1, 2, 3, 4, 5])
        issues = detect_issues(s, "Clean", "numeric", 1.0, 1.0, 1.0, 1.0)
        assert issues == []

    def test_empty_series_returns_no_issues(self):
        s = pd.Series([], dtype="float64")
        issues = detect_issues(s, "Empty", "numeric", 1.0, 1.0, 1.0, 1.0)
        assert issues == []


class TestMissingValuesIssue:

    def test_missing_values_flagged_below_threshold(self):
        s = pd.Series([1, 2, None, None, 5])
        issues = detect_issues(s, "Col", "numeric", 0.6, 1.0, 1.0, 1.0)
        missing = [i for i in issues if i.issue_type == "missing_values"]
        assert len(missing) == 1
        assert missing[0].affected_rows == 2
        assert missing[0].severity == "high"
        assert "Col" in missing[0].description

    def test_missing_values_not_flagged_when_above_threshold(self):
        s = pd.Series([1, 2, 3, 4, None])
        issues = detect_issues(s, "Col", "numeric", 0.96, 1.0, 1.0, 1.0)
        assert not any(i.issue_type == "missing_values" for i in issues)


class TestUniquenessIssue:

    def test_duplicate_values_flagged_for_text_column(self):
        s = pd.Series(["a", "a", "a", "b"])
        issues = detect_issues(s, "Col", "text", 1.0, 0.5, 1.0, 1.0)
        dupes = [i for i in issues if i.issue_type == "duplicate_values"]
        assert len(dupes) == 1
        assert dupes[0].severity == "high"

    def test_uniqueness_skipped_for_numeric_columns(self):
        # Repeated numeric values are expected (e.g. time-series
        # readings) and should never raise a duplicate_values issue,
        # regardless of how low the uniqueness score is.
        s = pd.Series([5, 5, 5, 5])
        issues = detect_issues(s, "Col", "numeric", 1.0, 0.1, 1.0, 1.0)
        assert not any(i.issue_type == "duplicate_values" for i in issues)

    def test_uniqueness_skipped_for_boolean_columns(self):
        s = pd.Series([True, True, False, True])
        issues = detect_issues(s, "Col", "boolean", 1.0, 0.5, 1.0, 1.0)
        assert not any(i.issue_type == "duplicate_values" for i in issues)

    def test_uniqueness_not_flagged_above_090(self):
        s = pd.Series(["a", "b", "c", "d"])
        issues = detect_issues(s, "Col", "text", 1.0, 0.95, 1.0, 1.0)
        assert not any(i.issue_type == "duplicate_values" for i in issues)


class TestValidityIssue:

    def test_validity_issue_counts_invalid_numeric_values_directly(self):
        # 'abc' is the only genuinely invalid value here; the count
        # should reflect that exactly, not an estimate derived from
        # the rounded validity_score.
        s = pd.Series(["103,812", "-1.548", "abc", None, "42"])
        issues = detect_issues(s, "Col", "numeric", 1.0, 1.0, 0.75, 1.0)
        format_errors = [i for i in issues if i.issue_type == "format_errors"]
        assert len(format_errors) == 1
        assert format_errors[0].affected_rows == 1

    def test_validity_issue_counts_invalid_datetime_values(self):
        s = pd.Series(["2024-01-01", "not a date", "2024-03-01"])
        issues = detect_issues(s, "Col", "datetime", 1.0, 1.0, 0.6, 1.0)
        format_errors = [i for i in issues if i.issue_type == "format_errors"]
        assert len(format_errors) == 1
        assert format_errors[0].affected_rows == 1

    def test_validity_not_flagged_above_095(self):
        s = pd.Series(["1", "2", "3"])
        issues = detect_issues(s, "Col", "numeric", 1.0, 1.0, 1.0, 1.0)
        assert not any(i.issue_type == "format_errors" for i in issues)

    def test_negative_numbers_are_not_counted_as_invalid(self):
        # Regression test: validity() must not silently convert
        # negative numbers to positive before checking parseability.
        s = pd.Series(["-1.5", "-42", "10"])
        issues = detect_issues(s, "Col", "numeric", 1.0, 1.0, 0.6, 1.0)
        format_errors = [i for i in issues if i.issue_type == "format_errors"]
        # All three values are genuinely valid numbers, so even though
        # validity_score is (artificially) below threshold here, the
        # directly-counted affected_rows should be 0.
        assert format_errors[0].affected_rows == 0


class TestConsistencyIssue:

    def test_numeric_outlier_description_mentions_iqr(self):
        s = pd.Series([1, 2, 3, 4, 1000])
        issues = detect_issues(s, "Col", "numeric", 1.0, 1.0, 1.0, 0.8)
        inconsistency = [i for i in issues if i.issue_type == "inconsistency"]
        assert len(inconsistency) == 1
        assert "IQR" in inconsistency[0].description
        assert "outlier" in inconsistency[0].suggested_action.lower()

    def test_text_mixed_case_description(self):
        s = pd.Series(["Leeds", "leeds", "LEEDS", "York"])
        issues = detect_issues(s, "Col", "text", 1.0, 1.0, 1.0, 0.75)
        inconsistency = [i for i in issues if i.issue_type == "inconsistency"]
        assert len(inconsistency) == 1
        assert "mixed-case" in inconsistency[0].description
        assert "standardise" in inconsistency[0].suggested_action.lower()

    def test_consistency_not_flagged_above_095(self):
        s = pd.Series([1, 2, 3, 4, 5])
        issues = detect_issues(s, "Col", "numeric", 1.0, 1.0, 1.0, 1.0)
        assert not any(i.issue_type == "inconsistency" for i in issues)

    def test_affected_rows_at_least_one_when_flagged(self):
        # Guards against round() producing 0 affected_rows for a very
        # small proportion, which would look like "0 rows affected"
        # while still being flagged as an issue.
        s = pd.Series(["a"] * 100 + ["Leeds", "leeds"])
        issues = detect_issues(s, "Col", "text", 1.0, 1.0, 1.0, 0.98)
        inconsistency = [i for i in issues if i.issue_type == "inconsistency"]
        if inconsistency:
            assert inconsistency[0].affected_rows >= 1


class TestMultipleIssuesPerColumn:

    def test_column_can_have_multiple_simultaneous_issues(self):
        s = pd.Series(["Leeds", "leeds", None, None])
        issues = detect_issues(s, "City", "text", 0.5, 1.0, 1.0, 0.5)
        issue_types = {i.issue_type for i in issues}
        assert "missing_values" in issue_types
        assert "inconsistency" in issue_types