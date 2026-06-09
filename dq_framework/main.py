from __future__ import annotations
import argparse
import sys

from src.profiling.pipeline import run
from src.reporting.generator import generate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Visual Analytics for Data Quality Assessment"
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to input dataset (CSV, Excel, JSON)")
    parser.add_argument("--output", "-o", default="outputs/reports",
                        help="Directory to write the report")
    args = parser.parse_args()

    print(f"[1/3] Loading: {args.input}")
    try:
        profile = run(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[2/3] Profiling complete")
    print(f"      Rows: {profile.n_rows:,}  Columns: {profile.n_cols}")
    print(f"      Score: {profile.overall_score:.2%}")
    print(f"      Issues: {len(profile.all_issues)}")

    print(f"[3/3] Generating report → {args.output}/")
    report_path = generate(profile, args.output)
    print(f"\nDone! Open this file in your browser:")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()