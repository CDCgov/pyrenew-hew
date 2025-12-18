import argparse
import datetime as dt
import logging
import os
from pathlib import Path

from prep_data import process_and_save_loc_data
from prep_eval_data import save_eval_data

from pipelines.cli_utils import add_common_forecast_arguments
from pipelines.common_utils import (
    calculate_training_dates,
    create_hubverse_table,
    get_available_reports,
    load_credentials,
    load_nssp_data,
    parse_and_validate_report_date,
    plot_and_save_loc_forecast,
    run_r_script,
)
from pipelines.forecast_pyrenew import (
    generate_epiweekly_data,
)


def timeseries_ensemble_forecasts(
    model_dir: Path, n_forecast_days: int, n_samples: int, epiweekly: bool = False
) -> None:
    script_args = [
        "--model-dir",
        f"{model_dir}",
        "--n-forecast-days",
        f"{n_forecast_days}",
        "--n-samples",
        f"{n_samples}",
    ]
    if epiweekly:
        script_args.append("--epiweekly")
    run_r_script(
        "pipelines/forecast_timeseries_ensemble.R",
        script_args,
        function_name="timeseries_ensemble_forecasts",
    )
    return None


def main(
    disease: str,
    report_date: str,
    loc: str,
    facility_level_nssp_data_dir: Path | str,
    state_level_nssp_data_dir: Path | str,
    param_data_dir: Path | str,
    output_dir: Path | str,
    n_training_days: int,
    n_forecast_days: int,
    n_samples: int,
    model_letters: str,
    eval_data_path: Path,
    exclude_last_n_days: int = 0,
    credentials_path: Path | None = None,
    nhsn_data_path: Path | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if model_letters != "e":
        raise ValueError(
            "Only model_letters 'e' is supported for 'timeseries' model_family."
        )

    ensemble_model_name = f"ts_ensemble_{model_letters}"

    logger.info(
        "Starting single-location timeseries forecasting pipeline for "
        f"location {loc}, and report date {report_date}"
    )

    credentials_dict = load_credentials(credentials_path, logger)

    available_facility_level_reports = get_available_reports(
        facility_level_nssp_data_dir
    )
    available_loc_level_reports = get_available_reports(state_level_nssp_data_dir)

    report_date, loc_report_date = parse_and_validate_report_date(
        report_date,
        available_facility_level_reports,
        available_loc_level_reports,
        logger,
    )

    first_training_date, last_training_date = calculate_training_dates(
        report_date,
        n_training_days,
        exclude_last_n_days,
        logger,
    )

    facility_level_nssp_data, loc_level_nssp_data = load_nssp_data(
        report_date,
        loc_report_date,
        available_facility_level_reports,
        available_loc_level_reports,
        facility_level_nssp_data_dir,
        state_level_nssp_data_dir,
        logger,
    )

    model_batch_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )
    model_batch_dir = Path(output_dir, model_batch_dir_name)

    model_run_dir = Path(model_batch_dir, "model_runs", loc)
    ensemble_model_output_dir = Path(model_run_dir, ensemble_model_name)

    os.makedirs(model_run_dir, exist_ok=True)
    os.makedirs(ensemble_model_output_dir, exist_ok=True)

    logger.info(f"Processing {loc}")
    process_and_save_loc_data(
        loc_abb=loc,
        disease=disease,
        facility_level_nssp_data=facility_level_nssp_data,
        loc_level_nssp_data=loc_level_nssp_data,
        loc_level_nwss_data=None,
        report_date=report_date,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        save_dir=Path(ensemble_model_output_dir, "data"),
        logger=logger,
        credentials_dict=credentials_dict,
        nhsn_data_path=nhsn_data_path,
    )

    logger.info("Getting eval data...")
    save_eval_data(
        loc=loc,
        disease=disease,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        latest_comprehensive_path=eval_data_path,
        output_data_dir=Path(ensemble_model_output_dir, "data"),
        last_eval_date=report_date + dt.timedelta(days=n_forecast_days),
        credentials_dict=credentials_dict,
        nhsn_data_path=nhsn_data_path,
    )
    logger.info("Done getting eval data.")

    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly_data(Path(ensemble_model_output_dir, "data"))

    logger.info("Data preparation complete.")

    n_days_past_last_training = n_forecast_days + exclude_last_n_days

    logger.info("Performing timeseries ensemble forecasting")
    timeseries_ensemble_forecasts(
        ensemble_model_output_dir, n_days_past_last_training, n_samples, epiweekly=False
    )
    timeseries_ensemble_forecasts(
        ensemble_model_output_dir, n_days_past_last_training, n_samples, epiweekly=True
    )

    plot_and_save_loc_forecast(
        model_run_dir,
        n_days_past_last_training,
        timeseries_model_name=ensemble_model_name,
    )
    create_hubverse_table(Path(model_run_dir, ensemble_model_name))

    logger.info("Postprocessing complete.")

    logger.info(
        "Single-location timeseries pipeline complete "
        f"for location {loc}, and "
        f"report date {report_date}."
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create fit data for disease modeling."
    )

    # Add common arguments
    add_common_forecast_arguments(parser)

    # Add timeseries-specific arguments
    parser.add_argument(
        "--model-letters",
        type=str,
        default="e",
        help=(
            "Fit the model corresponding to the provided model letters "
            "(e.g. 'he', 'e', 'hew')."
        ),
        required=True,
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help=("Number of samples to draw (default: 1000)."),
    )

    args = parser.parse_args()
    main(**vars(args))
