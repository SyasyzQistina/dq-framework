from __future__ import annotations
import pandas as pd
from src.models import QualityIssue


def _severity(score: float, high_t: float, med_t: float) -> str:
    if score < high_t:
        return "high"
    if score < med_t:
        return "medium"
    return "low"


def detect_issues(
    series: pd.Series,
    col_name: str,
    inferred_type: str,
    completeness_score: float,
    uniqueness_score: float,
    validity_score: float,
    consistency_score: float,
) -> list[QualityIssue]:
    issues = []
    n = len(series)

    # Missing values
    if completeness_score < 0.95:
        n_missing = int(series.isna().sum())
        issues.append(QualityIssue(
            issue_type="missing_values",
            severity=_severity(completeness_score, 0.80, 0.95),
            affected_rows=n_missing,
            pct_affected=round(n_missing / n * 100, 2),
            description=f"'{col_name}' has {n_missing} missing values "
                        f"({round(n_missing/n*100,1)}% of rows).",
            suggested_action="Impute or remove rows depending on missingness level.",
        ))

    # Duplicates
    if uniqueness_score < 0.90 and inferred_type not in ("categorical", "boolean"):
        n_dupes = int(series.duplicated(keep=False).sum())
        issues.append(QualityIssue(
            issue_type="duplicate_values",
            severity=_severity(uniqueness_score, 0.70, 0.90),
            affected_rows=n_dupes,
            pct_affected=round(n_dupes / n * 100, 2),
            description=f"'{col_name}' contains duplicate values "
                        f"(uniqueness: {uniqueness_score:.2f}).",
            suggested_action="Check if duplicates are intentional.",
        ))

    # Format errors
    if validity_score < 0.95:
        n_invalid = int(round((1 - validity_score) * series.dropna().__len__()))
        issues.append(QualityIssue(
            issue_type="format_error",
            severity=_severity(validity_score, 0.80, 0.95),
            affected_rows=n_invalid,
            pct_affected=round(n_invalid / n * 100, 2),
            description=f"'{col_name}' has values that don't match "
                        f"expected type '{inferred_type}'.",
            suggested_action="Coerce to correct type and audit invalid values.",
        ))

    # Inconsistency
    if consistency_score < 0.95:
        n_incon = int(round((1 - consistency_score) * series.dropna().__len__()))
        issues.append(QualityIssue(
            issue_type="inconsistency",
            severity=_severity(consistency_score, 0.80, 0.95),
            affected_rows=n_incon,
            pct_affected=round(n_incon / n * 100, 2),
            description=f"'{col_name}' has outliers or mixed-case values.",
            suggested_action="Standardise casing or investigate outliers.",
        ))

    return issues