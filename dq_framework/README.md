# Automated Visual Analytics for Data Quality Assessment

**MSc Advanced Computer Science (Data Analytics)**  
**University of Leeds — 2026**  
**Student:** Syasya Qistina Binti Sheikh Kamar  
**Supervisor:** Professor Samuel Wilson 

---

## Overview

This project implements an automated, visualisation-driven framework for assessing and reporting the quality of open datasets. Users upload a dataset file (CSV, Excel, or JSON) through a web interface and receive a structured HTML quality report with six visualisation charts, a sortable column profiles table, and plain-English issue descriptions — all without requiring any programming expertise.

---

## Requirements

Before running the system, ensure you have the following installed:

- **Python 3.10 or higher** — [Download](https://www.python.org/downloads/)
- **pip** (included with Python)
- **Git** (to clone the repository)

To check your Python version:
```bash
python3 --version
```

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/SyasyzQistina/dq-framework.git
cd dq-framework
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
```

### Step 3 — Activate the virtual environment

**On macOS / Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, pandas, matplotlib, NumPy, Jinja2, openpyxl, and pytest.

---

## Running the System

### Start the web application

```bash
python app.py
```

You should see output similar to:
```
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

### Open the interface

Open your web browser and go to:
```
http://127.0.0.1:5000
```

### Upload a dataset

1. Click **Choose File** (or drag and drop a file onto the upload area)
2. Select any CSV, Excel (.xlsx), or JSON dataset file
3. Click **Generate Report**
4. Wait a few seconds while the report is generated
5. The quality report will open automatically in your browser
6. Click **⬇ Download Report** to save the report as a standalone HTML file
7. Click **← New Dataset** to return and upload another file

---

## Understanding the Report

The generated report contains four sections:

### Summary
Six metric cards showing:
- **Overall** — the mean quality score across all columns (0–100%)
- **Completeness** — proportion of non-missing values
- **Uniqueness** — proportion of unique values (non-numeric columns only)
- **Validity** — proportion of values matching the expected data type
- **Consistency** — absence of outliers (numeric) or mixed-case values (text)
- **Issues** — total number of detected quality issues

**Colour key:**
- 🔵 **Blue** — Good quality (score ≥ 0.95)
- 🟠 **Orange** — Warning (score 0.80–0.95)
- 🔴 **Red** — Poor quality (score < 0.80)
- ⚫ **Grey** — Informational (no quality threshold meaning)

### Visualisations
Six charts providing different perspectives on dataset quality:
1. **Missingness per Column** — heatmap of missing value percentages
2. **Quality Dimensions** — radar chart of the four dimension scores
3. **Column Quality Scores** — horizontal bar chart sorted by score
4. **Issues by Severity** — count of high, medium, and low severity issues
5. **Outlier Detection** — consistency score per numeric column
6. **Category Frequencies** — value distributions for text columns

### Column Profiles
An interactive table showing all quality metrics per column. You can:
- **Search** by column name using the search box
- **Filter** to show only columns with issues using the checkbox
- **Sort** by any column by clicking the column header
- **Hover** over issue badges to see full issue details

### Issues Detail
Plain-English descriptions of all high and medium severity issues with suggested remediation actions.

---

## Quality Dimension Formulae

| Dimension | Formula |
|-----------|---------|
| Completeness | `non_null_count ÷ total_rows` |
| Uniqueness | `unique_count ÷ non_null_count` (non-numeric only) |
| Validity | `valid_type_count ÷ non_null_count` |
| Consistency (numeric) | `1 − (outlier_count ÷ non_null_count)` using IQR × 3 |
| Consistency (text) | `1 − (inconsistencies ÷ unique_count)` using lowercase normalisation |
| Column score | `(completeness + uniqueness + validity + consistency) ÷ 4` |
| Dataset score | `sum(column_scores) ÷ number_of_columns` |

**Severity thresholds:**
- **High** — score < 0.80 (more than 20% of rows affected)
- **Medium** — score 0.80–0.95 (5–20% of rows affected)
- **Low** — score ≥ 0.95 (fewer than 5% of rows affected)

---

## Running the Automated Tests

To run the full pytest test suite:

```bash
pytest tests/ -v
```

To run with coverage report:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Expected output: all tests pass with ≥ 80% coverage across core modules.

---

## Project Structure

```
dq_framework/
├── app.py                          # Flask web application (two routes)
├── main.py                         # Command-line entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── sample_data/                    # Sample datasets for testing
│   ├── air_quality.csv
│   ├── lcc_business_rates.csv
│   └── late_payment_interest.csv
├── src/
│   ├── models.py                   # DatasetProfile, ColumnProfile, QualityIssue
│   ├── ingestion/
│   │   ├── loader.py               # Multi-format loading, encoding detection
│   │   └── schema_detector.py      # Automatic data type inference
│   ├── profiling/
│   │   ├── dimensions.py           # Four quality dimension calculations
│   │   ├── issue_detector.py       # Issue detection and severity classification
│   │   └── pipeline.py             # Orchestration of all profiling steps
│   ├── visualisation/
│   │   └── charts.py               # Six chart types (matplotlib)
│   └── reporting/
│       ├── generator.py            # HTML report generation (Jinja2)
│       └── templates/
│           └── report.html.jinja2  # Report template
├── templates/
│   └── index.html                  # Upload interface
└── tests/                          # Automated test suite (pytest)
    ├── test_dimensions.py
    ├── test_issue_detector.py
    ├── test_loader.py
    ├── test_pipeline.py
    └── test_schema_detector.py
```

---

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Auto-detects encoding (UTF-8, Latin-1, Windows-1252, ISO-8859-1) |
| Excel | `.xlsx` | Reads first sheet; requires openpyxl |
| JSON | `.json` | Flat JSON arrays |

**File size limit:** 50 MB per upload.

---

## Troubleshooting

**Port already in use:**
```bash
kill $(lsof -t -i:5000) && python app.py
```

**ModuleNotFoundError:**  
Make sure the virtual environment is activated (`source venv/bin/activate`) before running.

**File upload fails:**  
Check that the file is in CSV, Excel, or JSON format and is under 50 MB.

**Charts not showing in downloaded report:**  
The report is self-contained — all charts are embedded as base64 images. If charts are missing, try regenerating the report in Chrome or Firefox.

**Encoding error on CSV file:**  
The system tries four encodings automatically. If all fail, try opening the file in Excel and re-saving as CSV (UTF-8).

---

## Stopping the Application

Press `CTRL + C` in the terminal to stop the Flask server.

To deactivate the virtual environment:
```bash
deactivate
```

---

## GitHub Repository

```
https://github.com/SyasyzQistina/dq-framework
```

Branch: `develop` (active development)  
Branch: `main` (stable releases)

Continuous integration is configured via GitHub Actions — all tests run automatically on every push.

---

## Contact

**Syasya Qistina Binti Sheikh Kamar**  
MSc Advanced Computer Science (Data Analytics)  
University of Leeds — 2026