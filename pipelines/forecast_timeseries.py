import argparse
import logging
import os
import shutil
from datetime import timedelta
from pathlib import Path

from prep_data import process_and_save_loc_data
from prep_eval_data import save_eval_data

from pipelines.cli_utils import add_common_forecast_arguments
from pipelines.common_utils import (
    calculate_training_dates,
    get_available_reports,
    load_credentials,
    load_nssp_data,
    parse_and_validate_report_date,
    run_r_script,
)
from pipelines.forecast_pyrenew import (
    generate_epiweekly_data,
)
from pipelines.prep_data import process_and_save_loc_data
from pipelines.prep_eval_data import save_eval_data


def plot_and_save_loc_forecast(
    model_run_dir: Path,
    n_forecast_days: int,
    timeseries_model_name: str,
) -> None:
    run_r_script(
        "pipelines/plot_and_save_loc_forecast.R",
        [
            f"{model_run_dir}",
            "--n-forecast-days",
            f"{n_forecast_days}",
            "--timeseries-model-name",
            f"{timeseries_model_name}",
        ],
        function_name="plot_and_save_loc_forecast",
    )
    return None


def timeseries_ensemble_forecasts(
    model_dir: Path, n_forecast_days: int, n_samples: int
) -> None:
    run_r_script(
        "pipelines/forecast_timeseries_ensemble.R",
        [
            "--model-dir",
            f"{model_dir}",
            "--n-forecast-days",
            f"{n_forecast_days}",
            "--n-samples",
            f"{n_samples}",
        ],
        function_name="timeseries_ensemble_forecasts",
    )
    return None


def cdc_flat_baseline_forecasts(model_dir: str, n_forecast_days: int) -> None:
    run_r_script(
        "pipelines/forecast_cdc_flat_baseline.R",
        [
            "--model-dir",
            f"{model_dir}",
            "--n-forecast-days",
            f"{n_forecast_days}",
        ],
        function_name="cdc_flat_baseline_forecasts",
    )
    return None


def create_hubverse_table(model_fit_path):
    run_r_script(
        "-e",
        [
            f"""
            forecasttools::write_tabular(
            hewr::model_fit_dir_to_hub_q_tbl('{model_fit_path}'),
            fs::path('{model_fit_path}', "hubverse_table", ext = "parquet")
            )
            """,
        ],
        function_name="create_hubverse_table",
        text=True,
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
    n_denominator_samples: int,
    model_letters: str,
    exclude_last_n_days: int = 0,
    eval_data_path: Path = None,
    credentials_path: Path = None,
    nhsn_data_path: Path = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if model_letters != "e":
        raise ValueError(
            "Only model_letters 'e' is supported for 'timeseries' model_family."
        )

    ensemble_model_name = f"ts_ensemble_{model_letters}"
    baseline_model_name = f"baseline_cdc_{model_letters}"

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
    baseline_model_output_dir = Path(model_run_dir, baseline_model_name)
    ensemble_model_output_dir = Path(model_run_dir, ensemble_model_name)

    os.makedirs(model_run_dir, exist_ok=True)
    os.makedirs(baseline_model_output_dir, exist_ok=True)
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
        save_dir=Path(baseline_model_output_dir, "data"),
        logger=logger,
        credentials_dict=credentials_dict,
        nhsn_data_path=nhsn_data_path,
    )

    logger.info("Getting eval data...")
    if eval_data_path is None:
        raise ValueError("No path to an evaluation dataset provided.")
    save_eval_data(
        loc=loc,
        disease=disease,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        latest_comprehensive_path=eval_data_path,
        output_data_dir=Path(baseline_model_output_dir, "data"),
        last_eval_date=report_date + timedelta(days=n_forecast_days),
        credentials_dict=credentials_dict,
        nhsn_data_path=nhsn_data_path,
    )
    logger.info("Done getting eval data.")

    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly_data(Path(baseline_model_output_dir, "data"))

    logger.info("Copying data from baseline to ensemble directory...")
    shutil.copytree(
        Path(baseline_model_output_dir, "data"), Path(ensemble_model_output_dir, "data")
    )

    logger.info("Data preparation complete.")

    logger.info("Performing baseline forecasting and postprocessing...")

    n_days_past_last_training = n_forecast_days + exclude_last_n_days
    cdc_flat_baseline_forecasts(baseline_model_output_dir, n_days_past_last_training)

    create_hubverse_table(Path(model_run_dir, baseline_model_name))

    logger.info("Performing timeseries ensemble forecasting")
    timeseries_ensemble_forecasts(
        ensemble_model_output_dir,
        n_days_past_last_training,
        n_denominator_samples,
    )
    plot_and_save_loc_forecast(
        model_run_dir, n_days_past_last_training, ensemble_model_name
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
        help=(
            "Number of posterior samples to draw per chain using NUTS (default: 1000)."
        ),
    )

    args = parser.parse_args()
    n_denominator_samples = args.n_samples * args.n_chains
    delattr(args, "n_samples")
    delattr(args, "n_chains")
    main(**vars(args), n_denominator_samples=n_denominator_samples)
