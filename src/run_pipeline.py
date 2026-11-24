"""
src/run_pipeline.py
Unified CLI entry point for the NFL Big Data Bowl 2027 Pass Rush Kinematic Index (PRKI) pipeline.

Usage:
    # Fast reproduction from serialized results (no raw 2.3 GB CSVs required):
    python src/run_pipeline.py

    # Full end-to-end execution from raw competition CSVs in data/:
    python src/run_pipeline.py --from-raw
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="NFL Big Data Bowl 2027 — Pass Rush Kinematic Index (PRKI) Pipeline"
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Run full ETL, kinematic extraction, econometric modeling, and figure generation from raw CSVs in data/.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also recompile writeup/nfl_big_data_bowl_2027_final_submission.pdf via headless Chrome/Edge.",
    )
    args = parser.parse_args()

    if args.from_raw:
        print("=== [1/4] Running Raw CSV -> Parquet ETL (src/etl.py) ===")
        import etl
        etl.main()

        print("\n=== [2/4] Running Econometric & Tracking Models (src/models.py) ===")
        import models
        models.run_all_models()

        print("\n=== [3/4] Generating 300 DPI Publication Figures (src/plots.py) ===")
        import plots
        plots.main()
    else:
        res_path = ROOT_DIR / "processed" / "model_results.json"
        if not res_path.exists():
            raise FileNotFoundError(
                f"Missing {res_path}. Run with --from-raw after placing competition CSVs in data/."
            )
        ct_path = ROOT_DIR / "processed" / "combine_tracking.parquet"
        if ct_path.exists():
            print("=== Regenerating 300 DPI Publication Figures (src/plots.py) ===")
            import plots
            plots.main()
        else:
            print(
                "Note: processed/combine_tracking.parquet not present locally; "
                "using pre-rendered 300 DPI figures in figures/ and authoritative processed/model_results.json."
            )

    print("\n=== Assembling Reproducible Kaggle Submission Notebook (src/build_notebook.py) ===")
    import build_notebook
    build_notebook.build_notebook()

    if args.pdf:
        print("\n=== Compiling 7-Page Publication PDF (src/md_to_pdf.py) ===")
        import md_to_pdf
        md_to_pdf.build_pdf()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
