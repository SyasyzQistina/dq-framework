import pathlib
import pytest
import pandas as pd
from src.profiling.pipeline import run
from src.reporting.generator import generate


@pytest.fixture
def sample_profile(tmp_path):
    df = pd.DataFrame({
        "id":   [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "age":  [25, 30, 22, 45, 33],
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return run(str(path))


class TestReportGenerator:
    def test_report_file_created(self, sample_profile, tmp_path):
        path = generate(sample_profile, str(tmp_path), make_charts=False)
        assert pathlib.Path(path).exists()

    def test_report_is_html(self, sample_profile, tmp_path):
        path = generate(sample_profile, str(tmp_path), make_charts=False)
        content = pathlib.Path(path).read_text()
        assert "<!DOCTYPE html>" in content

    def test_report_contains_dataset_name(self, sample_profile, tmp_path):
        path = generate(sample_profile, str(tmp_path), make_charts=False)
        content = pathlib.Path(path).read_text()
        assert sample_profile.dataset_name in content

    def test_report_contains_score(self, sample_profile, tmp_path):
        path = generate(sample_profile, str(tmp_path), make_charts=False)
        content = pathlib.Path(path).read_text()
        assert "Overall score" in content