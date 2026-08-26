"""HTML report generation.

Renders a DatasetProfile into a self-contained HTML report using the
Jinja2 template at templates/report.html.jinja2. Charts are generated
via visualisation/charts.py and base64-embedded directly into the
HTML (see _encode_image), so the resulting file has no external
dependencies and works fully offline once downloaded (FR18).
"""

from __future__ import annotations

import base64
import pathlib

from jinja2 import Environment, FileSystemLoader

from src.models import DatasetProfile
from src.visualisation.charts import generate_all

TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"


def _encode_image(path: str) -> str:
    """Convert an image file to a base64 data URI."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def generate(profile: DatasetProfile, output_dir: str,
             make_charts: bool = True) -> str:
    """Generate the HTML report for a profiled dataset.

    Writes the report to {output_dir}/{profile.dataset_name}_report.html
    and returns that path. If make_charts is True (the default), the
    six quality charts are generated fresh and base64-embedded into
    profile.generated_charts before rendering; set to False to reuse
    charts already attached to the profile.
    """
    out = pathlib.Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if make_charts:
        charts_dir = out / "charts"
        chart_paths = generate_all(profile, str(charts_dir))
        # Embed charts as base64 so they work in downloaded HTML,
        # with no dependency on the charts/ folder existing alongside it.
        profile.generated_charts = [
            _encode_image(p) for p in chart_paths
        ]

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("report.html.jinja2")
    html = template.render(profile=profile)

    report_path = out / f"{profile.dataset_name}_report.html"
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)