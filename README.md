# Automated Visual Analytics for Data Quality Assessment

MSc Advanced Computer Science (Data Analytics) — University of Leeds  
Supervisor: Professor Roy A. Ruddle

## Overview

An automated, visualisation-driven framework for assessing and reporting the quality of open datasets, built on [Ruddle's 6-step data quality method](https://github.com/royruddle/6-step-data-quality-method).

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/dq_framework.git
cd dq_framework
pip install -r requirements.txt

# 2. Run on a dataset
python main.py --input datasets/sample/example.csv --output outputs/reports/

# 3. Open the report
open outputs/reports/example_report.html
```

## Project structure

```
dq_framework/
├── src/
│   ├── models.py                  # DatasetProfile, ColumnProfile, QualityIssue
│   ├── ingestion/                 # Load & normalise any tabular format
│   ├── profiling/                 # 6-step quality pipeline
│   ├── visualisation/             # Chart generation (matplotlib)
│   └── reporting/                 # HTML/JSON report generation (Jinja2)
├── tests/                         # pytest test suite
├── datasets/
│   ├── raw/                       # Full evaluation datasets (gitignored)
│   └── sample/                    # Small fixtures for testing
├── outputs/                       # Generated reports & charts (gitignored)
├── notebooks/                     # Exploratory & evaluation notebooks
├── main.py                        # CLI entrypoint
└── requirements.txt
```

## Quality dimensions assessed

| Dimension | Description |
|---|---|
| Completeness | Fraction of non-null values per column |
| Uniqueness | Fraction of non-duplicate values |
| Validity | Values conforming to inferred data type |
| Consistency | Absence of outliers and formatting inconsistencies |

## Running tests

```bash
pytest tests/ -v --cov=src
```

## CI

GitHub Actions runs the full test suite on every push to `main` and `develop`.  
See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
