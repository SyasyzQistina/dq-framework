from __future__ import annotations
import pandas as pd
from src.models import QualityIssue


def _severity(score: float) -> str:
    if score < 0.80: return "high"
    if score < 0.95: return "medium"
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
    if n == 0:
        return issues

    # ── Completeness ──────────────────────────────────────────────
    if completeness_score < 0.95:
        n_missing = int(series.isna().sum())
        issues.append(QualityIssue(
            issue_type="missing_values",
            severity=_severity(completeness_score),
            affected_rows=n_missing,
            pct_affected=round(n_missing / n * 100, 2),
            description=(
                f"'{col_name}' has {n_missing:,} missing values "
                f"({n_missing/n*100:.1f}% of rows)."
            ),
            suggested_action=(
                "Impute or remove rows depending on missingness level."
            ),
        ))

    # ── Uniqueness — non-numeric columns only ─────────────────────
    if (uniqueness_score < 0.90
            and inferred_type not in ("numeric", "boolean")):
        non_null = series.dropna()
        n_dupes  = int(non_null.duplicated(keep=False).sum())
        issues.append(QualityIssue(
            issue_type="duplicate_values",
            severity=_severity(uniqueness_score),
            affected_rows=n_dupes,
            pct_affected=round(n_dupes / n * 100, 2),
            description=(
                f"'{col_name}' contains duplicate values "
                f"(uniqueness: {uniqueness_score:.2f})."
            ),
            suggested_action="Check if duplicates are intentional.",
        ))

    # ── Validity ─────────────────────────────────────────────────
    if validity_score < 0.95:
        n_invalid = round((1 - validity_score) * series.dropna().shape[0])
        issues.append(QualityIssue(
            issue_type="format_errors",
            severity=_severity(validity_score),
            affected_rows=n_invalid,
            pct_affected=round(n_invalid / n * 100, 2),
            description=(
                f"'{col_name}' has {n_invalid:,} values that do not match "
                f"the expected {inferred_type} format."
            ),
            suggested_action=(
                "Check for mixed data types or unexpected formatting."
            ),
        ))

    # ── Consistency ───────────────────────────────────────────────
    if consistency_score < 0.95:
        if inferred_type == "numeric":
            description = (
                f"'{col_name}' has outlier values detected via IQR method "
                f"(consistency: {consistency_score:.2f})."
            )
            suggested_action = (
                "Investigate outliers — may indicate data entry errors "
                "or sensor faults."
            )
        else:
            description = (
                f"'{col_name}' has mixed-case or inconsistent value formats "
                f"(e.g. 'Leeds', 'leeds', 'LEEDS')."
            )
            suggested_action = (
                "Standardise casing and formatting across all values."
            )

        non_null     = series.dropna()
        n_affected   = max(1, round((1 - consistency_score) * len(non_null)))
        issues.append(QualityIssue(
            issue_type="inconsistency",
            severity=_severity(consistency_score),
            affected_rows=n_affected,
            pct_affected=round(n_affected / n * 100, 2),
            description=description,
            suggested_action=suggested_action,
        ))

    return issues