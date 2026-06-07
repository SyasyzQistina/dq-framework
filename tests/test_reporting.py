"""Tests for the report generator."""
import pathlib
import pytest

from src.reporting.generator import generate


class TestReportGenerator:
    def test_report_file_created(self, sample_profile, tmp_path):
        path = generate(sample_profile, str(tmp_path), generate_charts=False)
        assert pathlib.Path(path).exists()

    def test_report_is_html(self, sample_profile, tmp_path):
        path = generate(sample_profile, str(tmp_path), generate_charts=False)
        content = pathlib.Path(path).read_text()
        assert "<!DOCTYPE html>" in content
        assert sample_profile.dataset_name in content

    def test_json_summary_created(self, sample_profile, tmp_path):
        generate(sample_profile, str(tmp_path), generate_charts=False)
        json_path = tmp_path / f"{sample_profile.dataset_name}_summary.json"
        assert json_path.exists()

    def test_report_contains_scores(self, sample_profile, tmp_path):
        path = generate(sample_profile, str(tmp_path), generate_charts=False)
        content = pathlib.Path(path).read_text()
        # Overall score should appear in the report
        assert "Overall score" in content
