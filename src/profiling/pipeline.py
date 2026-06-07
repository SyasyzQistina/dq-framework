"""
Pipeline orchestrator: implements Ruddle's 6-step data quality method.

Step 1  – Understand the dataset (shape, schema, basic stats)
Step 2  – Identify potential issues per column
Step 3  – Measure quality dimensions (completeness, uniqueness, validity, consistency)
Step 4  – Quantify each dimension score per column
Step 5  – Rank issues by severity
Step 6  – Summarise findings into a DatasetProfile
"""
from __future__ import annotations
import pathlib
from datetime import datetime

import pandas as pd

from src.models import DatasetProfile, ColumnProfile
from src.ingestion.loader import load
from src.ingestion.schema_detector import detect_schema
from src.profiling import dimensions as dim
from src.profiling.issue_detector import detect_issues


def run(source: str) -> DatasetProfile:
    """
    Run the full 6-step pipeline on a dataset.

    Args:
        source: file path or URL to the dataset.

    Returns:
        A fully populated DatasetProfile.
    """

    # ── STEP 1: Understand the dataset ──────────────────────────────────────
    df = load(source)
    schema = detect_schema(df)
    dataset_name = pathlib.Path(source).stem if not source.startswith("http") else source

    profile = DatasetProfile(
        dataset_name=dataset_name,
        source_path=source,
        n_rows=len(df),
        n_cols=len(df.columns),
        run_timestamp=datetime.utcnow(),
    )

    # ── STEPS 2–5: Per-column profiling ─────────────────────────────────────
    for col in df.columns:
        series = df[col]
        inferred_type = schema[col]

        # Step 3 & 4: measure + quantify
        completeness_score = dim.completeness(series)
        uniqueness_score   = dim.uniqueness(series)
        validity_score     = dim.validity(series, inferred_type)
        consistency_score  = dim.consistency(series, inferred_type)

        # Descriptive stats
        non_null = series.dropna()
        top_values: dict[str, int] = {}
        value_min = value_max = value_mean = value_std = None

        if inferred_type == "numeric":
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric):
                value_min  = float(numeric.min())
                value_max  = float(numeric.max())
                value_mean = float(numeric.mean())
                value_std  = float(numeric.std())
        else:
            top_values = (
                non_null.astype(str)
                .value_counts()
                .head(10)
                .to_dict()
            )

        # Step 2 & 5: identify and rank issues
        issues = detect_issues(
            series=series,
            col_name=col,
            inferred_type=inferred_type,
            completeness_score=completeness_score,
            uniqueness_score=uniqueness_score,
            validity_score=validity_score,
            consistency_score=consistency_score,
        )

        col_profile = ColumnProfile(
            col_name=col,
            inferred_type=inferred_type,
            n_missing=int(series.isna().sum()),
            pct_missing=round(series.isna().mean() * 100, 2),
            n_unique=int(series.nunique()),
            pct_unique=round(series.nunique() / len(series) * 100 if len(series) else 0, 2),
            completeness=completeness_score,
            uniqueness=uniqueness_score,
            validity=validity_score,
            consistency=consistency_score,
            value_min=value_min,
            value_max=value_max,
            value_mean=value_mean,
            value_std=value_std,
            top_values=top_values,
            issues=sorted(
                issues,
                key=lambda i: {"high": 0, "medium": 1, "low": 2}[i.severity]
            ),
        )
        profile.columns.append(col_profile)

    # ── STEP 6: Summarise ────────────────────────────────────────────────────
    dimensions = ["completeness", "uniqueness", "validity", "consistency"]
    for d in dimensions:
        scores = [getattr(c, d) for c in profile.columns]
        profile.dimension_scores[d] = round(sum(scores) / len(scores), 4) if scores else 0.0

    n_issues = len(profile.all_issues)
    high = sum(1 for i in profile.all_issues if i.severity == "high")
    medium = sum(1 for i in profile.all_issues if i.severity == "medium")

    profile.summary_text = (
        f"Dataset '{dataset_name}' — {profile.n_rows:,} rows × {profile.n_cols} columns. "
        f"Overall quality score: {profile.overall_score:.2%}. "
        f"{n_issues} issues detected ({high} high, {medium} medium severity). "
        f"Weakest dimension: "
        f"{min(profile.dimension_scores, key=profile.dimension_scores.get)} "
        f"({min(profile.dimension_scores.values()):.2%})."
    )

    return profile
