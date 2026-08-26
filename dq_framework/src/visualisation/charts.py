"""Chart generation for the quality report.

Six matplotlib charts, saved as PNG files and later base64-embedded
into the self-contained HTML report (see reporting/generator.py).
Runs on the non-interactive Agg backend so it works server-side with
no display attached.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.models import DatasetProfile

# ── IBM Carbon accessibility palette ─────────────────────────────────────
# GOOD/WARN/BAD encode a quality threshold and are used wherever a chart
# shows a score (charts 2, 3, 4, 5). GREY is used for purely informational
# content that carries no quality threshold (charts 2's fill, and chart 6).
GOOD = "#0f62fe"   # blue   - good quality  (>= 0.95)
WARN = "#e07b00"   # orange - warning       (0.80-0.95)
BAD = "#da1e28"    # red    - poor quality  (< 0.80)
GREY = "#888780"   # grey   - informational, no threshold
DARK = "#2C2C2A"
MID = "#6B7280"

CHART_W = 7
CHART_H = 4.5

# ── Global font settings - minimum 11pt everywhere (NFR04) ───────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.7,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.fontsize": 11,
})


def _score_color(score: float) -> str:
    """Map a 0-1 quality score to its IBM Carbon threshold colour."""
    if score >= 0.95:
        return GOOD
    if score >= 0.80:
        return WARN
    return BAD


def _save(fig, output_dir: pathlib.Path, filename: str) -> str:
    """Save a matplotlib figure to disk and close it, returning the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return str(path)


# ── 1. Missingness heatmap ────────────────────────────────────────────────
def missingness_heatmap(profile: DatasetProfile, output_dir: str) -> str:
    """Single-column heatmap of % missing per column, turbo colormap.

    Turbo (continuous blue-to-red) is used instead of the IBM Carbon
    palette here since this chart encodes a continuous quantity, not
    a threshold-based score.
    """
    out = pathlib.Path(output_dir)
    cols = [c.col_name for c in profile.columns]
    pct = [c.pct_missing for c in profile.columns]
    n = len(cols)

    h = min(12, max(4.5, n * 0.5))
    label_fs = max(10, min(12, int(220 / n)))

    fig, ax = plt.subplots(figsize=(CHART_W, h))
    data = np.array(pct).reshape(-1, 1)
    im = ax.imshow(data, aspect="auto", cmap="turbo", vmin=0, vmax=100)

    ax.set_yticks(range(n))
    ax.set_yticklabels(cols, fontsize=label_fs)
    ax.set_xticks([])
    ax.set_title("1. Missingness per Column", loc="left", fontsize=13)
    ax.set_xlabel("← Complete       Missing →", fontsize=12, color=MID)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("% missing", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # Annotate each row with its exact % missing, but only below 30
    # columns - beyond that the labels overlap (see Section 4.10.5
    # / 6.2.4 of the dissertation for the known scalability limit).
    if n <= 30:
        for i, p in enumerate(pct):
            if p > 0:
                color = "white" if p > 50 else DARK
                ax.text(0, i, f"  {p:.0f}%", va="center",
                        fontsize=max(10, label_fs - 1), color=color)

    fig.tight_layout()
    return _save(fig, out, "1_missingness_heatmap.png")


# ── 2. Dimension radar - grey fill, colour-coded score dots ──────────────
def dimension_radar(profile: DatasetProfile, output_dir: str) -> str:
    """Polar chart of the four aggregated dimension scores.

    The filled polygon itself is neutral grey (informational shape);
    each vertex dot is colour-coded by its own score threshold so weak
    dimensions stand out at a glance.
    """
    out = pathlib.Path(output_dir)
    dims = list(profile.dimension_scores.keys())
    scores = list(profile.dimension_scores.values())

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    scores_plot = scores + scores[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(CHART_W, CHART_W),
                           subplot_kw=dict(polar=True))

    ax.plot(angles_plot, scores_plot, "-", linewidth=1.5,
            color=GREY, markersize=0)
    ax.fill(angles_plot, scores_plot, alpha=0.12, color=GREY)

    for angle, score in zip(angles, scores):
        dot_color = _score_color(score)
        ax.plot(angle, score, "o", markersize=10, color=dot_color, zorder=5)
        offset = 0.16 if score > 0.7 else -0.16
        ax.text(angle, min(score + offset, 0.95), f"{score:.2f}",
                ha="center", va="center",
                fontsize=12, fontweight="bold", color=dot_color)

    ax.set_thetagrids(np.degrees(angles), dims, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=11, color=MID)
    ax.set_rlabel_position(45)
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.set_title("2. Quality Dimensions", loc="center", pad=22, fontsize=13)

    fig.tight_layout(pad=3.0)
    return _save(fig, out, "2_dimension_radar.png")


# ── 3. Column scores bar ──────────────────────────────────────────────────
def column_scores_bar(profile: DatasetProfile, output_dir: str) -> str:
    """Horizontal bar chart of each column's overall score, sorted
    ascending so the most problematic columns appear first."""
    out = pathlib.Path(output_dir)
    pairs = sorted(
        [(c.col_name, c.overall_score) for c in profile.columns],
        key=lambda x: x[1]
    )
    cols = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]
    colors = [_score_color(s) for s in scores]
    n = len(cols)

    h = min(14, max(4.5, n * 0.45))
    label_fs = max(10, min(12, int(220 / n)))

    fig, ax = plt.subplots(figsize=(CHART_W, h))
    bars = ax.barh(cols, scores, color=colors, height=0.6,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1.28)
    ax.set_xlabel("Quality score", fontsize=12)
    ax.set_title("3. Column Quality Scores", loc="left", fontsize=13)
    ax.axvline(0.8, color="#BBBBBB", linestyle="--", linewidth=1, label="0.80")
    ax.axvline(0.95, color="#DDDDDD", linestyle=":", linewidth=1, label="0.95")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(cols, fontsize=label_fs)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center",
                fontsize=max(10, label_fs - 1), color=DARK)

    fig.tight_layout()
    return _save(fig, out, "3_column_scores_bar.png")


# ── 4. Issue severity summary ─────────────────────────────────────────────
def issue_severity_summary(profile: DatasetProfile, output_dir: str) -> str:
    """Vertical bar chart counting issues by severity level."""
    out = pathlib.Path(output_dir)
    from collections import Counter
    counts = Counter(i.severity for i in profile.all_issues)
    severities = ["high", "medium", "low"]
    labels = ["High", "Medium", "Low"]
    colours = [BAD, WARN, GOOD]
    values = [counts.get(s, 0) for s in severities]

    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    bars = ax.bar(labels, values, color=colours, width=0.4,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of issues", fontsize=12)
    ax.set_title("4. Issues by Severity", loc="left", fontsize=13)
    ax.set_ylim(0, max(values) * 1.35 + 1)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=12)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, str(v),
                ha="center", fontsize=13, fontweight="bold", color=DARK)

    fig.tight_layout()
    return _save(fig, out, "4_issue_severity.png")


# ── 5. Outlier detection (consistency, numeric columns only) ─────────────
def boxplots(profile: DatasetProfile, output_dir: str) -> str:
    """Bar chart of the consistency score per numeric column, with
    reference lines at the 0.95 and 0.80 thresholds.

    Returns None if the dataset has no numeric columns - the caller
    (generate_all) filters out None results before returning.
    """
    out = pathlib.Path(output_dir)
    numeric_cols = [c for c in profile.columns if c.inferred_type == "numeric"]
    if not numeric_cols:
        return None

    names = [c.col_name for c in numeric_cols]
    scores = [c.consistency for c in numeric_cols]
    colors = [_score_color(s) for s in scores]
    n = len(names)

    w = min(12, max(CHART_W, n * 1.0))
    label_fs = max(10, min(12, int(130 / n)))

    fig, ax = plt.subplots(figsize=(w, CHART_H))
    bars = ax.bar(names, scores, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.25)
    ax.set_title("5. Outlier Detection (Consistency)", loc="left", fontsize=13)
    ax.set_ylabel("Score", fontsize=12)
    ax.axhline(0.95, color="#BBBBBB", linestyle="--",
               linewidth=1, label="0.95 good")
    ax.axhline(0.80, color="#DDDDDD", linestyle=":",
               linewidth=1, label="0.80 warning")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=label_fs)

    rotation = 30 if n > 5 else 0
    ha = "right" if n > 5 else "center"
    plt.xticks(rotation=rotation, ha=ha, fontsize=label_fs)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                score + 0.02, f"{score:.2f}",
                ha="center", fontsize=max(10, label_fs),
                fontweight="bold", color=DARK)

    fig.tight_layout()
    return _save(fig, out, "5_boxplots.png")


# ── 6. Category frequencies - neutral grey, no quality threshold ─────────
def frequency_plots(profile: DatasetProfile, output_dir: str) -> str:
    """Top-6 value frequency bar charts for up to 2 text/categorical
    columns. Grey throughout, since frequency counts carry no quality
    threshold meaning.

    Returns None if there are no eligible categorical/text columns.
    """
    out = pathlib.Path(output_dir)
    cat_cols = [c for c in profile.columns
                if c.inferred_type in ("categorical", "text")
                and c.top_values]
    if not cat_cols:
        return None

    n = min(2, len(cat_cols))
    subset = cat_cols[:n]

    fig, axes = plt.subplots(1, n, figsize=(CHART_W, CHART_H))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(subset):
        ax = axes[i]
        items = list(col.top_values.items())[:6]
        labels = [str(k)[:16] for k, _ in items]
        values = [v for _, v in items]

        bars = ax.barh(labels, values, color=GREY, alpha=0.85,
                       edgecolor="white", linewidth=0.4)
        ax.set_title(col.col_name[:20], fontsize=12, fontweight="bold")
        ax.set_xlabel("Count", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        for bar, v in zip(bars, values):
            ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{v:,}", va="center", fontsize=11, color=DARK)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("6. Category Frequencies",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, out, "6_frequency_plots.png")


# ── Main entry point ──────────────────────────────────────────────────────
def generate_all(profile: DatasetProfile, output_dir: str) -> list[str]:
    """Generate all six charts and return the file paths of those that
    were actually produced (boxplots/frequency_plots may return None
    if the dataset has no eligible columns for that chart type)."""
    charts = [
        missingness_heatmap(profile, output_dir),
        dimension_radar(profile, output_dir),
        column_scores_bar(profile, output_dir),
        issue_severity_summary(profile, output_dir),
        boxplots(profile, output_dir),
        frequency_plots(profile, output_dir),
    ]
    return [c for c in charts if c is not None]