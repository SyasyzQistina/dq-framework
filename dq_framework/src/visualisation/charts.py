from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from src.models import DatasetProfile

# ── Global style ────────────────────────────────────────────────────────────
TEAL   = "#1D9E75"
AMBER  = "#EF9F27"
RED    = "#E24B4A"
GREY   = "#F5F5F3"
DARK   = "#2C2C2A"
MID    = "#888780"

CHART_W = 10   # every chart the same width (inches)
CHART_H = 4.5  # every chart the same height

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "x",
    "grid.color":        "#EEEEEE",
    "grid.linewidth":    0.6,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})


def _score_color(score: float) -> str:
    if score >= 0.95: return TEAL
    if score >= 0.80: return AMBER
    return RED


def _save(fig, output_dir: pathlib.Path, filename: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return str(path)


# ── 1. Missingness heatmap ───────────────────────────────────────────────────
def missingness_heatmap(profile: DatasetProfile, output_dir: str) -> str:
    out  = pathlib.Path(output_dir)
    cols = [c.col_name for c in profile.columns]
    pct  = [c.pct_missing for c in profile.columns]
    n    = len(cols)

    fig, ax = plt.subplots(figsize=(CHART_W, max(CHART_H, n * 0.38)))
    data = np.array(pct).reshape(-1, 1)
    im   = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)

    ax.set_yticks(range(n))
    ax.set_yticklabels(cols, fontsize=9)
    ax.set_xticks([])
    ax.set_title("1. Missing Values per Column (%)", loc="left")
    ax.set_xlabel("← Complete          Missing →", fontsize=9, color=MID)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("% missing", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Annotate each bar
    for i, p in enumerate(pct):
        label = f"{p:.1f}%" if p > 0 else "0%"
        color = "white" if p > 50 else DARK
        ax.text(0, i, f"  {label}", va="center", fontsize=8, color=color)

    fig.tight_layout()
    return _save(fig, out, "1_missingness_heatmap.png")


# ── 2. Dimension radar ───────────────────────────────────────────────────────
def dimension_radar(profile: DatasetProfile, output_dir: str) -> str:
    out    = pathlib.Path(output_dir)
    dims   = list(profile.dimension_scores.keys())
    scores = list(profile.dimension_scores.values())

    angles      = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    scores_plot = scores + scores[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H + 1),
                           subplot_kw=dict(polar=True))

    ax.plot(angles_plot, scores_plot, "o-", linewidth=2,
            color=TEAL, markersize=6)
    ax.fill(angles_plot, scores_plot, alpha=0.18, color=TEAL)

    ax.set_thetagrids(np.degrees(angles), dims, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=MID)
    ax.set_rlabel_position(45)
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.set_title("2. Quality Dimensions Overview", loc="center", pad=20)

    # Score labels on each point
    for angle, score in zip(angles, scores):
        ax.text(angle, score + 0.07, f"{score:.2f}",
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=TEAL)

    fig.tight_layout()
    return _save(fig, out, "2_dimension_radar.png")


# ── 3. Column scores bar ─────────────────────────────────────────────────────
def column_scores_bar(profile: DatasetProfile, output_dir: str) -> str:
    out    = pathlib.Path(output_dir)
    cols   = [c.col_name for c in profile.columns]
    scores = [c.overall_score for c in profile.columns]
    colors = [_score_color(s) for s in scores]
    n      = len(cols)

    fig, ax = plt.subplots(figsize=(CHART_W, max(CHART_H, n * 0.42)))
    bars = ax.barh(cols, scores, color=colors, height=0.6,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Overall quality score")
    ax.set_title("3. Per-Column Quality Scores", loc="left")
    ax.axvline(0.8, color="#AAAAAA", linestyle="--",
               linewidth=1, label="0.80 threshold")
    ax.axvline(0.95, color="#CCCCCC", linestyle=":",
               linewidth=1, label="0.95 threshold")
    ax.legend(fontsize=8, loc="lower right")

    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=8, color=DARK)

    fig.tight_layout()
    return _save(fig, out, "3_column_scores_bar.png")


# ── 4. Issue severity ────────────────────────────────────────────────────────
def issue_severity_summary(profile: DatasetProfile, output_dir: str) -> str:
    out = pathlib.Path(output_dir)
    from collections import Counter
    counts     = Counter(i.severity for i in profile.all_issues)
    severities = ["high", "medium", "low"]
    labels     = ["High", "Medium", "Low"]
    colours    = [RED, AMBER, TEAL]
    values     = [counts.get(s, 0) for s in severities]

    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    bars = ax.bar(labels, values, color=colours, width=0.45,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of issues")
    ax.set_title("4. Issues by Severity", loc="left")
    ax.set_ylim(0, max(values) * 1.3 + 1)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, str(v),
                ha="center", fontsize=11, fontweight="bold", color=DARK)

    fig.tight_layout()
    return _save(fig, out, "4_issue_severity.png")

# ── 6. Outlier / consistency bar ─────────────────────────────────────────────
def boxplots(profile: DatasetProfile, output_dir: str) -> str:
    out          = pathlib.Path(output_dir)
    numeric_cols = [c for c in profile.columns if c.inferred_type == "numeric"]
    if not numeric_cols:
        return None

    names  = [c.col_name for c in numeric_cols]
    scores = [c.consistency for c in numeric_cols]
    colors = [_score_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    bars = ax.bar(names, scores, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Consistency score (1.0 = no outliers)")
    ax.set_title("6. Outlier Detection — Consistency per Numeric Column",
                 loc="left")
    ax.axhline(0.95, color="#AAAAAA", linestyle="--",
               linewidth=1, label="0.95 good threshold")
    ax.axhline(0.80, color="#CCCCCC", linestyle=":",
               linewidth=1, label="0.80 warning threshold")
    ax.legend(fontsize=8)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    plt.xticks(rotation=30, ha="right", fontsize=9)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                score + 0.02, f"{score:.2f}",
                ha="center", fontsize=9, fontweight="bold", color=DARK)

    fig.tight_layout()
    return _save(fig, out, "6_boxplots.png")


# ── 7. Frequency plots ───────────────────────────────────────────────────────
def frequency_plots(profile: DatasetProfile, output_dir: str) -> str:
    out      = pathlib.Path(output_dir)
    cat_cols = [c for c in profile.columns
                if c.inferred_type in ("categorical", "text")
                and c.top_values]
    if not cat_cols:
        return None

    n      = min(6, len(cat_cols))
    subset = cat_cols[:n]
    cols_n = min(3, n)
    rows_n = (n + cols_n - 1) // cols_n

    fig, axes = plt.subplots(rows_n, cols_n,
                              figsize=(CHART_W, CHART_H * rows_n))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(subset):
        ax     = axes[i]
        items  = list(col.top_values.items())[:8]
        labels = [str(k)[:18] for k, _ in items]
        values = [v for _, v in items]

        bars = ax.barh(labels, values, color=TEAL, alpha=0.8,
                       edgecolor="white", linewidth=0.4)
        ax.set_title(col.col_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Count", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        for bar, v in zip(bars, values):
            ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=8, color=DARK)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("7. Categorical Column Frequencies",
                 fontsize=12, fontweight="bold", x=0, ha="left", y=1.01)
    fig.tight_layout()
    return _save(fig, out, "7_frequency_plots.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def generate_all(profile: DatasetProfile, output_dir: str) -> list[str]:
    charts = [
        missingness_heatmap(profile, output_dir),
        dimension_radar(profile, output_dir),
        column_scores_bar(profile, output_dir),
        issue_severity_summary(profile, output_dir),
        boxplots(profile, output_dir),
        frequency_plots(profile, output_dir),
    ]
    return [c for c in charts if c is not None]