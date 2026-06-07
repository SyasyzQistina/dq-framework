"""
CLI entrypoint for the DQ Framework.

Usage:
    python main.py --input datasets/raw/mydata.csv --output outputs/reports/
    python main.py --input https://example.com/data.csv --output outputs/reports/
"""
import argparse
import sys
import pathlib

from src.profiling.pipeline import run
from src.reporting.generator import generate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Visual Analytics for Data Quality Assessment"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path or URL to the input dataset (CSV, Excel, JSON, Parquet)"
    )
    parser.add_argument(
        "--output", "-o", default="outputs/reports",
        help="Directory to write the HTML report and charts (default: outputs/reports)"
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation (faster, useful for quick checks)"
    )
    args = parser.parse_args()

    print(f"[1/3] Loading dataset: {args.input}")
    try:
        profile = run(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[2/3] Profiling complete — {profile.n_rows:,} rows, {profile.n_cols} columns")
    print(f"      Overall score: {profile.overall_score:.2%}")
    print(f"      Issues found:  {len(profile.all_issues)}")

    print(f"[3/3] Generating report → {args.output}/")
    report_path = generate(
        profile=profile,
        output_dir=args.output,
        generate_charts=not args.no_charts,
    )
    print(f"\nDone! Report saved to: {report_path}")
    print(f"Summary: {profile.summary_text}")


if __name__ == "__main__":
    main()
