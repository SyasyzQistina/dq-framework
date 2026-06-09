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

    # STEP 1: Understand the dataset
    df = load(source)
    schema = detect_schema(df)
    dataset_name = pathlib.Path(source).stem

    profile = DatasetProfile(
        dataset_name=dataset_name,
        source_path=source,
        n_rows=len(df),
        n_cols=len(df.columns),
        run_timestamp=datetime.now(),
    )

    # STEPS 2-5: Profile each column
    for col in df.columns:
        series = df[col]
        inferred_type = schema[col]

        # Step 3 & 4: measure dimensions
        completeness_score = dim.completeness(series)
        uniqueness_score   = dim.uniqueness(series)
        validity_score     = dim.validity(series, inferred_type)
        consistency_score  = dim.consistency(series, inferred_type)

        # Descriptive stats
        top_values = {}
        if inferred_type == "numeric":
            numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        else:
            top_values = (
                series.dropna().astype(str)
                .value_counts().head(5).to_dict()
            )

        # Step 2 & 5: detect and rank issues
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
            pct_unique=round(series.nunique() / len(series) * 100, 2),
            completeness=completeness_score,
            uniqueness=uniqueness_score,
            validity=validity_score,
            consistency=consistency_score,
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
        profile.dimension_scores[d] = round(sum(scores) / len(scores), 4)

    high = sum(1 for i in profile.all_issues if i.severity == "high")
    medium = sum(1 for i in profile.all_issues if i.severity == "medium")
    weakest = min(profile.dimension_scores, key=profile.dimension_scores.get)

    profile.summary_text = (
        f"Dataset '{dataset_name}' — {profile.n_rows:,} rows × "
        f"{profile.n_cols} columns. "
        f"Overall score: {profile.overall_score:.2%}. "
        f"{len(profile.all_issues)} issues found "
        f"({high} high, {medium} medium). "
        f"Weakest dimension: {weakest} "
        f"({profile.dimension_scores[weakest]:.2%})."
    )

    return profile