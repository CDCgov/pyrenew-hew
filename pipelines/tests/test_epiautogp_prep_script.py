"""
Script to test EpiAutoGP data preparation workflow.

This script replicates the data setup steps from `pipelines/forecast_pyrenew.py`
up to the point where `data_for_model_fit.json` is created, then uses
that file to create the EpiAutoGP input JSON in the same directory.
"""

import datetime as dt
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is in the path
sys.path.insert(0, ".")


from pipelines.common_utils import (
    calculate_training_dates,
    get_available_reports,
    load_nssp_data,
    parse_and_validate_report_date,
)
from pipelines.epiautogp import convert_to_epiautogp_json
from pipelines.prep_data import process_and_save_loc_data


def main():
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

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test parameters matching forecast_pyrenew.py test setup
    report_date_str = "2024-12-21"
    report_date = dt.date(2024, 12, 21)
    n_training_days = 90
    exclude_last_n_days = 0

    logger.info(
        f"Testing EpiAutoGP data prep for {disease}, {location}, target={target}"
    )

    # Set up data directories (matching forecast_pyrenew.py)
    facility_level_nssp_data_dir = base_dir / "private_data" / "nssp_etl_gold"
    state_level_nssp_data_dir = base_dir / "private_data" / "nssp_state_level_gold"
    nhsn_data_path = (
        base_dir / "private_data" / "nhsn_test_data" / f"{disease}_{location}.parquet"
    )

    # Get available reports and validate report date
    available_facility_level_reports = get_available_reports(
        facility_level_nssp_data_dir
    )
    available_loc_level_reports = get_available_reports(state_level_nssp_data_dir)

    report_date, loc_report_date = parse_and_validate_report_date(
        report_date_str,
        available_facility_level_reports,
        available_loc_level_reports,
        logger,
    )

    # Calculate training dates
    first_training_date, last_training_date = calculate_training_dates(
        report_date,
        n_training_days,
        exclude_last_n_days,
        logger,
    )

    # Load NSSP data
    facility_level_nssp_data, loc_level_nssp_data = load_nssp_data(
        report_date,
        loc_report_date,
        available_facility_level_reports,
        available_loc_level_reports,
        facility_level_nssp_data_dir,
        state_level_nssp_data_dir,
        logger,
    )

    # Set up output directory structure (matching forecast_pyrenew.py)
    model_batch_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )

    output_dir = base_dir / f"{report_date}_epiautogp_test"
    model_batch_dir = output_dir / model_batch_dir_name
    model_run_dir = model_batch_dir / "model_runs" / location

    # For EpiAutoGP test, we'll create a simple model directory
    model_name = f"epiautogp_{target}"
    model_dir = model_run_dir / model_name
    data_dir = model_dir / "data"
    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Processing {location} data...")
    logger.info(f"Data will be saved to {data_dir}")

    # Process and save location data - this creates data_for_model_fit.json
    process_and_save_loc_data(
        loc_abb=location,
        disease=disease,
        facility_level_nssp_data=facility_level_nssp_data,
        loc_level_nssp_data=loc_level_nssp_data,
        loc_level_nwss_data=None,  # No wastewater for this test
        report_date=report_date,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        save_dir=data_dir,
        logger=logger,
        credentials_dict={},
        nhsn_data_path=nhsn_data_path,
    )

    # Verify data_for_model_fit.json was created
    data_for_model_fit_path = data_dir / "data_for_model_fit.json"
    if not data_for_model_fit_path.exists():
        print(
            f"ERROR: data_for_model_fit.json not created at {data_for_model_fit_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info(f"Successfully created {data_for_model_fit_path}")

    # Now use that data_for_model_fit.json to create EpiAutoGP input
    # Place it in the same data directory
    epiautogp_json_path = data_dir / f"epiautogp_input_{target}.json"

    try:
        result_path = convert_to_epiautogp_json(
            data_for_model_fit_path=data_for_model_fit_path,
            output_json_path=epiautogp_json_path,
            disease=disease,
            location=location,
            forecast_date=report_date,
            target=target,
            logger=logger,
        )
        logger.info(f"Successfully created EpiAutoGP input at {result_path}")

        # Verify both files are in the same directory
        assert result_path.parent == data_for_model_fit_path.parent, (
            f"EpiAutoGP JSON not in same directory as data_for_model_fit.json: "
            f"{result_path.parent} != {data_for_model_fit_path.parent}"
        )

        logger.info(
            f"Verified: Both data_for_model_fit.json and epiautogp_input_{target}.json "
            f"are in {data_dir}"
        )

        # Also copy to the requested output location for the test script
        import shutil

        output_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(result_path, output_json)
        logger.info(f"Also copied to {output_json} for test verification")

        print(f"SUCCESS: Created {epiautogp_json_path}")
        print(f"SUCCESS: Verified files are in same directory: {data_dir}")

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
