"""
Script to test EpiAutoGP data preparation workflow.

This script uses the refactored forecast_utils functions to prepare data
and create the EpiAutoGP input JSON.
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is in the path
sys.path.insert(0, ".")


from pipelines.epiautogp import convert_to_epiautogp_json
from pipelines.forecast_utils import prepare_model_data, setup_forecast_pipeline


def main():
    if len(sys.argv) not in [6, 7, 8]:
        print(
            "Usage: python test_epiautogp_prep_script.py <target> <disease> "
            "<location> <base_dir> <output_json> [<frequency> [<use_percentage>]]",
            file=sys.stderr,
        )
        sys.exit(1)

    target = sys.argv[1]
    disease = sys.argv[2]
    location = sys.argv[3]
    base_dir = Path(sys.argv[4])
    output_json = Path(sys.argv[5])
    frequency = sys.argv[6] if len(sys.argv) >= 7 else "epiweekly"
    use_percentage = sys.argv[7].lower() == "true" if len(sys.argv) >= 8 else False

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test parameters
    report_date_str = "2024-12-21"
    n_training_days = 90
    n_forecast_days = 28
    exclude_last_n_days = 0

    logger.info(
        f"Testing EpiAutoGP data prep for {disease}, {location}, target={target}"
    )

    # Set up data directories
    facility_level_nssp_data_dir = base_dir / "private_data" / "nssp_etl_gold"
    state_level_nssp_data_dir = base_dir / "private_data" / "nssp_state_level_gold"
    nhsn_data_path = (
        base_dir / "private_data" / "nhsn_test_data" / f"{disease}_{location}.parquet"
    )
    eval_data_path = state_level_nssp_data_dir / f"{report_date_str}.parquet"
    output_dir = base_dir / f"{report_date_str}_epiautogp_test"

    # Model name for EpiAutoGP
    model_name = f"epiautogp_{target}"
    if use_percentage:
        model_name += "_pct"

    # Step 1: Setup forecast pipeline context
    logger.info("Setting up forecast pipeline...")
    context = setup_forecast_pipeline(
        disease=disease,
        report_date=report_date_str,
        loc=location,
        facility_level_nssp_data_dir=facility_level_nssp_data_dir,
        state_level_nssp_data_dir=state_level_nssp_data_dir,
        output_dir=output_dir,
        n_training_days=n_training_days,
        n_forecast_days=n_forecast_days,
        exclude_last_n_days=exclude_last_n_days,
        credentials_path=None,
        logger=logger,
    )

    # Step 2: Prepare data (process location data, eval data, epiweekly data)
    logger.info("Preparing model data...")
    paths = prepare_model_data(
        context=context,
        model_name=model_name,
        eval_data_path=eval_data_path,
        nhsn_data_path=nhsn_data_path,
    )

    logger.info(f"Data directory: {paths.data_dir}")
    logger.info(f"Daily training data: {paths.daily_training_data}")
    logger.info(f"Epiweekly training data: {paths.epiweekly_training_data}")

    # Step 3: Convert to EpiAutoGP JSON format
    logger.info("Converting to EpiAutoGP JSON format...")
    epiautogp_json_path = paths.data_dir / f"epiautogp_input_{target}.json"

    try:
        result_path = convert_to_epiautogp_json(
            daily_training_data_path=paths.daily_training_data,
            epiweekly_training_data_path=paths.epiweekly_training_data,
            output_json_path=epiautogp_json_path,
            disease=disease,
            location=location,
            forecast_date=context.report_date,
            target=target,
            frequency=frequency,
            use_percentage=use_percentage,
            logger=logger,
        )
        logger.info(f"Successfully created EpiAutoGP input at {result_path}")

        # Verify the file was created in the data directory
        assert result_path.parent == paths.data_dir, (
            f"EpiAutoGP JSON not in data directory: {result_path.parent} != {paths.data_dir}"
        )

        logger.info(f"Verified: epiautogp_input_{target}.json is in {paths.data_dir}")

        # Also copy to the requested output location for the test script
        import shutil

        output_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(result_path, output_json)
        logger.info(f"Also copied to {output_json} for test verification")

        print(f"SUCCESS: Created {epiautogp_json_path}")
        print(f"SUCCESS: Verified files are in same directory: {paths.data_dir}")

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
