"""
Data models for the DQ framework.
DatasetProfile -> ColumnProfile -> QualityIssue
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class QualityIssue:
    """A single detected data quality problem within a column."""
    issue_type: str          # e.g. "missing_values", "duplicate_rows", "outlier"
    severity: str            # "high" | "medium" | "low"
    affected_rows: int
    pct_affected: float
    description: str
    suggested_action: str


@dataclass
class ColumnProfile:
    """Quality profile for a single column (steps 2–5)."""
    col_name: str
    inferred_type: str       # "numeric" | "categorical" | "datetime" | "text" | "boolean"

    # Missingness
    n_missing: int = 0
    pct_missing: float = 0.0

    # Uniqueness
    n_unique: int = 0
    pct_unique: float = 0.0

    # Quality dimension scores [0.0 = worst, 1.0 = best]
    completeness: float = 1.0
    uniqueness: float = 1.0
    validity: float = 1.0
    consistency: float = 1.0

    # Descriptive stats (populated for numeric columns)
    value_min: float | None = None
    value_max: float | None = None
    value_mean: float | None = None
    value_std: float | None = None

    # Top values for categorical columns
    top_values: dict[str, int] = field(default_factory=dict)

    # Detected issues for this column
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        return round(
            (self.completeness + self.uniqueness + self.validity + self.consistency) / 4, 4
        )


@dataclass
class DatasetProfile:
    """Top-level quality report for one dataset run (steps 1 & 6)."""
    dataset_name: str
    source_path: str
    n_rows: int
    n_cols: int
    run_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Per-dimension aggregate scores across all columns
    dimension_scores: dict[str, float] = field(default_factory=dict)

    # All column profiles
    columns: list[ColumnProfile] = field(default_factory=list)

    # Paths to generated chart files
    generated_charts: list[str] = field(default_factory=list)

    # Step 6 plain-text summary
    summary_text: str = ""

    @property
    def overall_score(self) -> float:
        if not self.columns:
            return 0.0
        return round(sum(c.overall_score for c in self.columns) / len(self.columns), 4)

    @property
    def all_issues(self) -> list[QualityIssue]:
        return [issue for col in self.columns for issue in col.issues]

    def get_column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.col_name == name), None)
