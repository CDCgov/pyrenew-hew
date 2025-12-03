#!/usr/bin/env python3
"""
Step 2: Transform PyRenew data to EpiAutoGP format.

This script takes the data_for_model_fit.json created in Step 1 and converts it
to the JSON format expected by the EpiAutoGP Julia model.

Usage:
    python pipelines/epiautogp_tests/step2_transform_to_epiautogp.py
"""

from pathlib import Path

from pipelines.prep_epiautogp_data import convert_to_epiautogp_format

# Configuration - must match Step 1
TEST_OUTPUT_DIR = Path("pipelines/epiautogp_tests/test_output")
TEST_LOCATION = "CA"
TEST_DISEASE = "COVID-19"
MAX_TRAIN_DATE_STR = "2024-12-21"

# Input and output paths
INPUT_JSON_PATH = (
    TEST_OUTPUT_DIR
    / "bootstrap_private_data"
    / TEST_LOCATION
    / "data"
    / "data_for_model_fit.json"
)
OUTPUT_JSON_PATH = TEST_OUTPUT_DIR / "epiautogp_input.json"


def main():
    """Transform PyRenew data to EpiAutoGP format."""
    print("=" * 70)
    print("EpiAutoGP Integration Test - Step 2: Transform to EpiAutoGP Format")
    print("=" * 70)

    # Verify input file exists
    if not INPUT_JSON_PATH.exists():
        print(f"\n✗ ERROR: Input file not found: {INPUT_JSON_PATH}")
        print("\nPlease run Step 1 first:")
        print("  python pipelines/epiautogp_tests/test_epiautogp_integration.py")
        return 1

    print(f"\nInput file: {INPUT_JSON_PATH}")
    print(f"Output file: {OUTPUT_JSON_PATH}")
    print("\nTarget: NHSN (hospital admissions)")
    print(f"Disease: {TEST_DISEASE}")
    print(f"Location: {TEST_LOCATION}")
    print(f"Forecast date: {MAX_TRAIN_DATE_STR}")

    try:
        print("\n" + "=" * 70)
        print("Converting PyRenew data to EpiAutoGP format...")
        print("=" * 70 + "\n")

        # Convert the data
        epiautogp_input = convert_to_epiautogp_format(
            input_json_path=str(INPUT_JSON_PATH),
            output_json_path=str(OUTPUT_JSON_PATH),
            target="nhsn",
            disease=TEST_DISEASE,
            location=TEST_LOCATION,
            forecast_date=MAX_TRAIN_DATE_STR,
            nowcast_reports_path=None,  # No nowcast for minimal test
        )

        print("\n" + "=" * 70)
        print("✓ Step 2 COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nEpiAutoGP input JSON created: {OUTPUT_JSON_PATH}")
        print("\nData summary:")
        print(f"  - Number of historical dates: {len(epiautogp_input['dates'])}")
        print(f"  - Number of reports: {len(epiautogp_input['reports'])}")
        print(
            f"  - Date range: {min(epiautogp_input['dates'])} to {max(epiautogp_input['dates'])}"
        )
        print(f"  - Forecast date: {epiautogp_input['forecast_date']}")
        print(f"  - Nowcast dates: {len(epiautogp_input['nowcast_dates'])}")
        print(f"  - Target: {epiautogp_input['target']}")
        print(f"  - Pathogen: {epiautogp_input['pathogen']}")
        print(f"  - Location: {epiautogp_input['location']}")

        print("\nNext steps:")
        print("  3. Run Julia model with this input")
        print("  4. Verify outputs")

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Step 2 FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
