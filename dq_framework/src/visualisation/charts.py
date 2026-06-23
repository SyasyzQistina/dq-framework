from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.models import DatasetProfile

# ── Accessible colour scheme (blue/orange — colour-blind friendly) ───────────
GOOD = "#2166AC"   # blue   — good quality
WARN = "#F4A736"   # orange — warning
BAD  = "#D6604D"   # red-orange — bad
DARK = "#2C2C2A"
MID  = "#888780"

# ── Consistent chart dimensions ──────────────────────────────────────────────
CHART_W = 5
CHART_H = 3.8

# ── Global font standardisation ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "x",
    "grid.color":        "#EEEEEE",
    "grid.linewidth":    0.7,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "legend.fontsize":   9,
})


def _score_color(score: float) -> str:
    if score >= 0.95: return GOOD
    if score >= 0.80: return WARN
    return BAD


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

    h = min(6, max(3, n * 0.4))
    fig, ax = plt.subplots(figsize=(CHART_W, h))

    data = np.array(pct).reshape(-1, 1)
    im   = ax.imshow(data, aspect="auto", cmap="turbo", vmin=0, vmax=100)

    ax.set_yticks(range(n))
    ax.set_yticklabels(cols, fontsize=10)
    ax.set_xticks([])
    ax.set_title("1. Missingness per Column", loc="left", fontsize=11)
    ax.set_xlabel("← Complete    Missing →", fontsize=10, color=MID)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% missing", fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    for i, p in enumerate(pct):
        label = f"{p:.0f}%"
        color = "white" if p > 50 else DARK
        ax.text(0, i, f"  {label}", va="center", fontsize=10, color=color)

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

    fig, ax = plt.subplots(figsize=(CHART_W, CHART_W),
                           subplot_kw=dict(polar=True))

    ax.plot(angles_plot, scores_plot, "o-", linewidth=2,
            color=GOOD, markersize=6)
    ax.fill(angles_plot, scores_plot, alpha=0.18, color=GOOD)

    # Add padding so labels don't clip
    ax.set_thetagrids(np.degrees(angles), dims, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=9, color=MID)
    ax.set_rlabel_position(45)
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.set_title("2. Quality Dimensions",
                 loc="center", pad=22, fontsize=11)

    for angle, score in zip(angles, scores):
        ax.text(angle, score + 0.12, f"{score:.2f}",
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=GOOD)

    fig.tight_layout(pad=1.5)
    return _save(fig, out, "2_dimension_radar.png")


# ── 3. Column scores bar ─────────────────────────────────────────────────────
def column_scores_bar(profile: DatasetProfile, output_dir: str) -> str:
    out    = pathlib.Path(output_dir)
    cols   = [c.col_name for c in profile.columns]
    scores = [c.overall_score for c in profile.columns]
    colors = [_score_color(s) for s in scores]
    n      = len(cols)

    h = min(6, max(3, n * 0.38))
    fig, ax = plt.subplots(figsize=(CHART_W, h))

    bars = ax.barh(cols, scores, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("Quality score", fontsize=10)
    ax.set_title("3. Column Quality Scores", loc="left", fontsize=11)
    ax.axvline(0.8, color="#BBBBBB", linestyle="--",
               linewidth=1, label="0.80")
    ax.axvline(0.95, color="#DDDDDD", linestyle=":",
               linewidth=1, label="0.95")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_yticklabels(cols, fontsize=10)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=9, color=DARK)

    fig.tight_layout()
    return _save(fig, out, "3_column_scores_bar.png")


# ── 4. Issue severity ────────────────────────────────────────────────────────
def issue_severity_summary(profile: DatasetProfile, output_dir: str) -> str:
    out = pathlib.Path(output_dir)
    from collections import Counter
    counts     = Counter(i.severity for i in profile.all_issues)
    severities = ["high", "medium", "low"]
    labels     = ["High", "Medium", "Low"]
    colours    = [BAD, WARN, GOOD]
    values     = [counts.get(s, 0) for s in severities]

    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    bars = ax.bar(labels, values, color=colours, width=0.4,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of issues", fontsize=10)
    ax.set_title("4. Issues by Severity", loc="left", fontsize=11)
    ax.set_ylim(0, max(values) * 1.35 + 1)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=10)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, str(v),
                ha="center", fontsize=11, fontweight="bold", color=DARK)

    fig.tight_layout()
    return _save(fig, out, "4_issue_severity.png")


# ── 5. Outlier / consistency bar ─────────────────────────────────────────────
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
    ax.set_ylim(0, 1.22)
    # Use title instead of rotated y-axis label
    ax.set_title("5. Outlier Detection (Consistency)",
                 loc="left", fontsize=11)
    ax.set_ylabel("Score", fontsize=10)
    ax.axhline(0.95, color="#BBBBBB", linestyle="--",
               linewidth=1, label="0.95 good")
    ax.axhline(0.80, color="#DDDDDD", linestyle=":",
               linewidth=1, label="0.80 warning")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=20, ha="right", fontsize=10)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                score + 0.02, f"{score:.2f}",
                ha="center", fontsize=10, fontweight="bold", color=DARK)

    fig.tight_layout()
    return _save(fig, out, "5_boxplots.png")


# ── 6. Frequency plots ───────────────────────────────────────────────────────
def frequency_plots(profile: DatasetProfile, output_dir: str) -> str:
    out      = pathlib.Path(output_dir)
    cat_cols = [c for c in profile.columns
                if c.inferred_type in ("categorical", "text")
                and c.top_values]
    if not cat_cols:
        return None

    # Show max 2 columns side by side for readability
    n      = min(2, len(cat_cols))
    subset = cat_cols[:n]

    fig, axes = plt.subplots(1, n, figsize=(CHART_W, CHART_H))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(subset):
        ax     = axes[i]
        items  = list(col.top_values.items())[:6]
        labels = [str(k)[:14] for k, _ in items]
        values = [v for _, v in items]

        bars = ax.barh(labels, values, color=GOOD, alpha=0.85,
                       edgecolor="white", linewidth=0.4)
        ax.set_title(col.col_name[:16], fontsize=10, fontweight="bold")
        ax.set_xlabel("Count", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        for bar, v in zip(bars, values):
            ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=10, color=DARK)

    fig.suptitle("6. Category Frequencies",
                 fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, out, "6_frequency_plots.png")


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