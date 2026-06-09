from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.models import DatasetProfile


def _save(fig, output_dir: pathlib.Path, filename: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def missingness_heatmap(profile: DatasetProfile, output_dir: str) -> str:
    out = pathlib.Path(output_dir)
    cols = [c.col_name for c in profile.columns]
    pct  = [c.pct_missing for c in profile.columns]

    fig, ax = plt.subplots(figsize=(10, max(3, len(cols) * 0.4)))
    data = np.array(pct).reshape(-1, 1)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_xticks([])
    ax.set_title("Missing values per column (%)", fontsize=12)
    plt.colorbar(im, ax=ax, label="% missing")
    return _save(fig, out, "missingness_heatmap.png")


def dimension_radar(profile: DatasetProfile, output_dir: str) -> str:
    out = pathlib.Path(output_dir)
    dims   = list(profile.dimension_scores.keys())
    scores = list(profile.dimension_scores.values())

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    scores_plot = scores + scores[:1]
    angles      = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores_plot, "o-", linewidth=2, color="#1D9E75")
    ax.fill(angles, scores_plot, alpha=0.2, color="#1D9E75")
    ax.set_thetagrids(np.degrees(angles[:-1]), dims, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Quality dimensions — {profile.dataset_name}", fontsize=12, pad=20)
    return _save(fig, out, "dimension_radar.png")


def column_scores_bar(profile: DatasetProfile, output_dir: str) -> str:
    out = pathlib.Path(output_dir)
    cols   = [c.col_name for c in profile.columns]
    scores = [c.overall_score for c in profile.columns]

    cmap    = plt.cm.RdYlGn
    colours = [cmap(s) for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(3, len(cols) * 0.45)))
    bars = ax.barh(cols, scores, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Overall quality score")
    ax.set_title("Per-column quality scores", fontsize=12)
    ax.axvline(0.8, color="#555", linestyle="--", linewidth=0.8, label="0.80 threshold")
    ax.legend(fontsize=8)
    return _save(fig, out, "column_scores_bar.png")


def issue_severity_summary(profile: DatasetProfile, output_dir: str) -> str:
    out = pathlib.Path(output_dir)
    from collections import Counter
    counts     = Counter(i.severity for i in profile.all_issues)
    severities = ["high", "medium", "low"]
    colours    = ["#E24B4A", "#EF9F27", "#639922"]
    values     = [counts.get(s, 0) for s in severities]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(severities, values, color=colours, width=0.5)
    ax.set_ylabel("Number of issues")
    ax.set_title("Issues by severity", fontsize=12)
    for bar, v in zip(bars, values):
        if v:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(v), ha="center", fontsize=10
            )
    return _save(fig, out, "issue_severity.png")


def generate_all(profile: DatasetProfile, output_dir: str) -> list[str]:
    return [
        missingness_heatmap(profile, output_dir),
        dimension_radar(profile, output_dir),
        column_scores_bar(profile, output_dir),
        issue_severity_summary(profile, output_dir),
    ]