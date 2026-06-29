from __future__ import annotations
import pathlib
import re
from datetime import datetime
import pandas as pd

from src.models import DatasetProfile, ColumnProfile
from src.ingestion.loader import load
from src.ingestion.schema_detector import detect_schema
from src.profiling import dimensions as dim
from src.profiling.issue_detector import detect_issues


def _clean_name(stem: str) -> str:
    """Convert file stem to a clean readable dataset name.
    - Replace underscores and hyphens with spaces
    - Remove date patterns like 31-12-2025, 11_06_2026, 2026-06-11
    - Remove 'final', 'v1', 'v2' suffixes
    - Strip extra whitespace
    """
    name = stem.replace("_", " ").replace("-", " ")
    # Remove date patterns: DD MM YYYY or YYYY MM DD (with spaces after replacement)
    name = re.sub(r'\b\d{2} \d{2} \d{4}\b', '', name)
    name = re.sub(r'\b\d{4} \d{2} \d{2}\b', '', name)
    # Remove standalone 4-digit years
    name = re.sub(r'\b(19|20)\d{2}\b', '', name)
    # Remove common suffixes
    name = re.sub(r'\b(final|v\d+|with coordinates?|clean|raw)\b',
                  '', name, flags=re.IGNORECASE)
    # Clean up multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    # Title case
    return name.title() if name else stem


def run(source: str) -> DatasetProfile:

    # STEP 1: Load dataset
    df   = load(source)
    schema = detect_schema(df)
    dataset_name = _clean_name(pathlib.Path(source).stem)

    profile = DatasetProfile(
        dataset_name=dataset_name,
        source_path=source,
        n_rows=len(df),
        n_cols=len(df.columns),
        run_timestamp=datetime.now(),
    )

    # STEPS 2–5: Profile each column
    for col in df.columns:
        series        = df[col]
        inferred_type = schema[col]

        completeness_score = dim.completeness(series)
        uniqueness_score   = dim.uniqueness(series)
        validity_score     = dim.validity(series, inferred_type)
        consistency_score  = dim.consistency(series, inferred_type)

        # Descriptive stats
        top_values = {}
        value_min = value_max = value_mean = value_std = None

        if inferred_type == "numeric":
            cleaned = (series.dropna().astype(str)
                       .str.replace(",", "").str.strip())
            numeric = pd.to_numeric(cleaned, errors="coerce").dropna()
            if len(numeric) > 0:
                value_min  = float(numeric.min())
                value_max  = float(numeric.max())
                value_mean = float(numeric.mean())
                value_std  = float(numeric.std()) if len(numeric) > 1 else 0.0
        else:
            top_values = (
                series.dropna().astype(str)
                .value_counts().head(5).to_dict()
            )

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
            pct_unique=round(
                series.nunique() / len(series) * 100, 2
            ) if len(series) > 0 else 0.0,
            completeness=completeness_score,
            uniqueness=uniqueness_score,
            validity=validity_score,
            consistency=consistency_score,
            top_values=top_values,
            value_min=value_min,
            value_max=value_max,
            value_mean=value_mean,
            value_std=value_std,
            issues=sorted(
                issues,
                key=lambda i: {"high": 0, "medium": 1, "low": 2}[i.severity]
            ),
        )
        profile.columns.append(col_profile)

    # STEP 6: Summarise
    dimensions = ["completeness", "uniqueness", "validity", "consistency"]
    for d in dimensions:
        scores = [getattr(c, d) for c in profile.columns]
        profile.dimension_scores[d] = (
            round(sum(scores) / len(scores), 4) if scores else 0.0
        )

    high    = sum(1 for i in profile.all_issues if i.severity == "high")
    medium  = sum(1 for i in profile.all_issues if i.severity == "medium")
    weakest = min(profile.dimension_scores,
                  key=profile.dimension_scores.get)

    profile.summary_text = (
        f"{profile.n_rows:,} rows \u00d7 {profile.n_cols} columns \u00b7 "
        f"Score: {profile.overall_score:.2%} \u00b7 "
        f"{len(profile.all_issues)} issues ({high} high, {medium} medium) \u00b7 "
        f"Weakest: {weakest} ({profile.dimension_scores[weakest]:.2%})"
    )

    return profile