"""
Issue detection: translate dimension scores into QualityIssue records.
"""
from __future__ import annotations
import pandas as pd
from src.models import QualityIssue


SEVERITY_THRESHOLDS = {
    # (high_threshold, medium_threshold) — score BELOW these → that severity
    "completeness":  (0.80, 0.95),
    "uniqueness":    (0.70, 0.90),
    "validity":      (0.80, 0.95),
    "consistency":   (0.80, 0.95),
}


def _severity(score: float, dimension: str) -> str:
    high_t, med_t = SEVERITY_THRESHOLDS[dimension]
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
    issues: list[QualityIssue] = []
    n = len(series)

    # Missing values
    if completeness_score < 0.95:
        n_missing = series.isna().sum()
        issues.append(QualityIssue(
            issue_type="missing_values",
            severity=_severity(completeness_score, "completeness"),
            affected_rows=int(n_missing),
            pct_affected=round(n_missing / n * 100, 2),
            description=f"Column '{col_name}' has {n_missing} missing values "
                        f"({round(n_missing/n*100,1)}% of rows).",
            suggested_action="Impute with mean/median (numeric) or mode (categorical), "
                             "or flag and exclude rows if missingness is >50%.",
        ))

    # Duplicate values (low uniqueness)
    if uniqueness_score < 0.90 and inferred_type not in ("categorical", "boolean"):
        n_dupes = int(series.duplicated(keep=False).sum())
        issues.append(QualityIssue(
            issue_type="duplicate_values",
            severity=_severity(uniqueness_score, "uniqueness"),
            affected_rows=n_dupes,
            pct_affected=round(n_dupes / n * 100, 2),
            description=f"Column '{col_name}' contains duplicate values "
                        f"(uniqueness score: {uniqueness_score}).",
            suggested_action="Investigate whether duplicates are intentional; "
                             "deduplicate at row level if this is a key column.",
        ))

    # Validity issues
    if validity_score < 0.95:
        n_invalid = int(round((1 - validity_score) * series.dropna().__len__()))
        issues.append(QualityIssue(
            issue_type="format_error",
            severity=_severity(validity_score, "validity"),
            affected_rows=n_invalid,
            pct_affected=round(n_invalid / n * 100, 2),
            description=f"Column '{col_name}' has values that do not match "
                        f"expected type '{inferred_type}'.",
            suggested_action="Coerce to correct type and audit unparseable values.",
        ))

    # Consistency / outliers
    if consistency_score < 0.95:
        n_inconsistent = int(round((1 - consistency_score) * series.dropna().__len__()))
        issues.append(QualityIssue(
            issue_type="inconsistency",
            severity=_severity(consistency_score, "consistency"),
            affected_rows=n_inconsistent,
            pct_affected=round(n_inconsistent / n * 100, 2),
            description=f"Column '{col_name}' shows inconsistency "
                        f"(outliers or mixed-case variants detected).",
            suggested_action="Review outliers; standardise casing for categorical values.",
        ))

    return issues
