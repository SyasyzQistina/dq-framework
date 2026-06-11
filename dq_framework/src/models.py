from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QualityIssue:
    issue_type: str
    severity: str
    affected_rows: int
    pct_affected: float
    description: str
    suggested_action: str


@dataclass
class ColumnProfile:
    col_name: str
    inferred_type: str
    n_missing: int = 0
    pct_missing: float = 0.0
    n_unique: int = 0
    pct_unique: float = 0.0
    completeness: float = 1.0
    uniqueness: float = 1.0
    validity: float = 1.0
    consistency: float = 1.0
    top_values: dict[str, int] = field(default_factory=dict)
    value_min: float | None = None
    value_max: float | None = None
    value_mean: float | None = None
    value_std: float | None = None
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        return round(
            (self.completeness + self.uniqueness +
             self.validity + self.consistency) / 4, 4
        )


@dataclass
class DatasetProfile:
    dataset_name: str
    source_path: str
    n_rows: int
    n_cols: int
    run_timestamp: datetime = field(default_factory=datetime.now)
    dimension_scores: dict[str, float] = field(default_factory=dict)
    columns: list[ColumnProfile] = field(default_factory=list)
    generated_charts: list[str] = field(default_factory=list)
    summary_text: str = ""

    @property
    def overall_score(self) -> float:
        if not self.columns:
            return 0.0
        return round(
            sum(c.overall_score for c in self.columns) / len(self.columns), 4
        )

    @property
    def all_issues(self) -> list[QualityIssue]:
        return [i for col in self.columns for i in col.issues]

    def get_column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.col_name == name), None)