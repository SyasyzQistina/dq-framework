"""
Report generator: combine charts + DatasetProfile into an HTML report.
"""
from __future__ import annotations
import json
import pathlib
from dataclasses import asdict
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from src.models import DatasetProfile
from src.visualisation import charts


TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"


def generate(
    profile: DatasetProfile,
    output_dir: str,
    generate_charts: bool = True,
) -> str:
    """
    Generate an HTML report for the given profile.

    Args:
        profile:          A fully populated DatasetProfile.
        output_dir:       Directory to write the report and charts into.
        generate_charts:  Set False to skip chart generation (useful in tests).

    Returns:
        Absolute path to the generated HTML file.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate charts
    if generate_charts:
        chart_paths = charts.generate_all(profile, str(out / "charts"))
        profile.generated_charts = chart_paths

    # Render template
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

    def format_number(value: int) -> str:
        return f"{value:,}"

    env.filters["format_number"] = format_number

    template = env.get_template("report.html.jinja2")
    html = template.render(
        profile=profile,
        high_count=sum(1 for i in profile.all_issues if i.severity == "high"),
    )

    report_path = out / f"{profile.dataset_name}_report.html"
    report_path.write_text(html, encoding="utf-8")

    # Also save a JSON summary alongside the HTML
    summary_path = out / f"{profile.dataset_name}_summary.json"
    summary_path.write_text(
        json.dumps(asdict(profile), indent=2, default=str),
        encoding="utf-8",
    )

    return str(report_path)
