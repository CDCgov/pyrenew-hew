"""
Script to convert surveillance data to EpiAutoGP JSON format.

This script is called by test_epiautogp_prep.sh to convert NHSN or NSSP
data to the JSON format expected by EpiAutoGP.
"""

import sys
from pathlib import Path

# Ensure the project root is in the path
sys.path.insert(0, ".")

import datetime as dt

from pipelines.epiautogp import convert_to_epiautogp_json


def main():
    """Convert surveillance data to EpiAutoGP JSON format."""
    if len(sys.argv) != 6:
        print(
            "Usage: python test_epiautogp_prep_script.py <target> <disease> "
            "<location> <base_dir> <output_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    target = sys.argv[1]
    disease = sys.argv[2]
    location = sys.argv[3]
    base_dir = Path(sys.argv[4])
    output_json = Path(sys.argv[5])
    forecast_date = dt.date(2024, 12, 21)

    # Paths
    combined_data_path = (
        base_dir / "combined_training_data_dummy.tsv"
    )  # Placeholder for NHSN
    nhsn_path = (
        base_dir / "private_data" / "nhsn_test_data" / f"{disease}_{location}.parquet"
        if target == "nhsn"
        else None
    )

    # For NSSP, we would need combined_training_data.tsv from prep_data
    # For now, we only test NHSN since we have that data
    if target == "nssp":
        print(
            "SKIPPING: NSSP conversion requires running "
            "prep_data.process_and_save_loc_data first"
        )
        sys.exit(0)

    try:
        convert_to_epiautogp_json(
            combined_training_data_path=combined_data_path,
            nhsn_data_path=nhsn_path,
            output_json_path=output_json,
            disease=disease,
            location=location,
            forecast_date=forecast_date,
            target=target,
        )
        print(f"SUCCESS: Created {output_json}")
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
