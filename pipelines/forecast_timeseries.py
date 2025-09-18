import argparse
import logging
import os
import subprocess
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from pipelines.forecast_pyrenew import (
    generate_epiweekly_data,
    get_available_reports,
)
from pipelines.prep_data import process_and_save_loc_data
from pipelines.prep_eval_data import save_eval_data


def plot_and_save_loc_forecast(
    model_run_dir: Path,
    n_forecast_days: int,
    timeseries_model_name: str,
) -> None:
    command = [
        "Rscript",
        "pipelines/plot_and_save_loc_forecast.R",
        f"{model_run_dir}",
        "--n-forecast-days",
        f"{n_forecast_days}",
        "--timeseries-model-name",
        f"{timeseries_model_name}",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"plot_and_save_loc_forecast: {result.stderr.decode('utf-8')}"
        )
    return None


def timeseries_ensemble_forecasts(
    model_run_dir: Path, model_name: str, n_forecast_days: int, n_samples: int
) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/forecast_timeseries_ensemble.R",
            f"{model_run_dir}",
            "--model-name",
            f"{model_name}",
            "--n-forecast-days",
            f"{n_forecast_days}",
            "--n-samples",
            f"{n_samples}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"timeseries_ensemble_forecasts: {result.stderr.decode('utf-8')}"
        )
    return None


def cdc_flat_baseline_forecasts(
    model_run_dir: Path, model_name: str, n_forecast_days: int
) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/forecast_cdc_flat_baseline.R",
            f"{model_run_dir}",
            "--model-name",
            f"{model_name}",
            "--n-forecast-days",
            f"{n_forecast_days}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cdc_flat_baseline_forecasts: {result.stderr.decode('utf-8')}"
        )
    return None


def create_hubverse_table(model_fit_path):
    result = subprocess.run(
        [
            "Rscript",
            "-e",
            f"""
            forecasttools::write_tabular(
            hewr::model_fit_dir_to_hub_q_tbl('{model_fit_path}'),
            fs::path('{model_fit_path}', "hubverse_table", ext = "parquet")
            )
            """,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"create_hubverse_table: {result.stderr}")
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

    if credentials_path is not None:
        cp = Path(credentials_path)
        if not cp.suffix.lower() == ".toml":
            raise ValueError(
                "Credentials file must have the extension "
                "'.toml' (not case-sensitive). Got "
                f"{cp.suffix}"
            )
        logger.info(f"Reading in credentials from {cp}...")
        with open(cp, "rb") as fp:
            credentials_dict = tomllib.load(fp)
    else:
        logger.info("No credentials file given. Will proceed without one.")
        credentials_dict = None

    available_facility_level_reports = get_available_reports(
        facility_level_nssp_data_dir
    )
    available_loc_level_reports = get_available_reports(state_level_nssp_data_dir)
    first_available_loc_report = min(available_loc_level_reports)
    last_available_loc_report = max(available_loc_level_reports)

    if report_date == "latest":
        report_date = max(available_facility_level_reports)
    else:
        report_date = datetime.strptime(report_date, "%Y-%m-%d").date()

    if report_date in available_loc_level_reports:
        loc_report_date = report_date
    elif report_date > last_available_loc_report:
        loc_report_date = last_available_loc_report
    elif report_date > first_available_loc_report:
        raise ValueError(
            "Dataset appear to be missing some state-level "
            f"reports. First entry is {first_available_loc_report}, "
            f"last is {last_available_loc_report}, but no entry "
            f"for {report_date}"
        )
    else:
        raise ValueError(
            "Requested report date is earlier than the first "
            "state-level vintage. This is not currently supported"
        )

    logger.info(f"Report date: {report_date}")
    if loc_report_date is not None:
        logger.info(f"Using location-level data as of: {loc_report_date}")

    # + 1 because max date in dataset is report_date - 1
    last_training_date = report_date - timedelta(days=exclude_last_n_days + 1)

    if last_training_date >= report_date:
        raise ValueError(
            "Last training date must be before the report date. "
            f"Got a last training date of {last_training_date} "
            f"with a report date of {report_date}."
        )

    logger.info(f"last training date: {last_training_date}")

    first_training_date = last_training_date - timedelta(days=n_training_days - 1)

    logger.info(f"First training date {first_training_date}")

    facility_level_nssp_data, loc_level_nssp_data = None, None

    if report_date in available_facility_level_reports:
        logger.info("Facility level data available for the given report date")
        facility_datafile = f"{report_date}.parquet"
        facility_level_nssp_data = pl.scan_parquet(
            Path(facility_level_nssp_data_dir, facility_datafile)
        )
    if loc_report_date in available_loc_level_reports:
        logger.info("location-level data available for the given report date.")
        loc_datafile = f"{loc_report_date}.parquet"
        loc_level_nssp_data = pl.scan_parquet(
            Path(state_level_nssp_data_dir, loc_datafile)
        )
    if facility_level_nssp_data is None and loc_level_nssp_data is None:
        raise ValueError(
            f"No data available for the requested report date {report_date}"
        )

    model_batch_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )
    model_batch_dir = Path(output_dir, model_batch_dir_name)
    model_run_dir = Path(model_batch_dir, "model_runs", loc)

    os.makedirs(model_run_dir, exist_ok=True)

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
        model_run_dir=model_run_dir,
        logger=logger,
        credentials_dict=credentials_dict,
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
        output_data_dir=Path(model_run_dir, "data"),
        last_eval_date=report_date + timedelta(days=n_forecast_days),
        credentials_dict=credentials_dict,
    )
    logger.info("Done getting eval data.")

    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly_data(model_run_dir)

    logger.info("Data preparation complete.")

    logger.info("Performing baseline forecasting and postprocessing...")

    n_days_past_last_training = n_forecast_days + exclude_last_n_days
    cdc_flat_baseline_forecasts(
        model_run_dir, baseline_model_name, n_days_past_last_training
    )

    create_hubverse_table(Path(model_run_dir, baseline_model_name))

    logger.info("Performing timeseries ensemble forecasting")
    timeseries_ensemble_forecasts(
        model_run_dir,
        ensemble_model_name,
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
    parser.add_argument(
        "--disease",
        type=str,
        required=True,
        help="Disease to model (e.g., COVID-19, Influenza, RSV).",
    )

    parser.add_argument(
        "--loc",
        type=str,
        required=True,
        help=(
            "Two-letter USPS abbreviation for the location to fit"
            "(e.g. 'AK', 'AL', 'AZ', etc.)."
        ),
    )

    parser.add_argument(
        "--model-letters",
        type=str,
        default="e",
        help=(
            "Fit the model corresponding to the provided model letters (e.g. 'he', 'e', 'hew')."
        ),
        required=True,
    )

    parser.add_argument(
        "--credentials-path",
        type=Path,
        help=("Path to a TOML file containing credentials such as API keys."),
    )

    parser.add_argument(
        "--report-date",
        type=str,
        default="latest",
        help="Report date in YYYY-MM-DD format or latest (default: latest).",
    )

    parser.add_argument(
        "--facility-level-nssp-data-dir",
        type=Path,
        default=Path("private_data", "nssp_etl_gold"),
        help=("Directory in which to look for facility-level NSSP ED visit data"),
    )

    parser.add_argument(
        "--state-level-nssp-data-dir",
        type=Path,
        default=Path("private_data", "nssp_state_level_gold"),
        help=("Directory in which to look for state-level NSSP ED visit data."),
    )

    parser.add_argument(
        "--param-data-dir",
        type=Path,
        default=Path("private_data", "prod_param_estimates"),
        help=("Directory in which to look for parameter estimatessuch as delay PMFs."),
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="private_data",
        help="Directory in which to save output.",
    )

    parser.add_argument(
        "--n-training-days",
        type=int,
        default=180,
        help="Number of training days (default: 180).",
    )

    parser.add_argument(
        "--n-forecast-days",
        type=int,
        default=28,
        help=(
            "Number of days ahead to forecast relative to the "
            "report date (default: 28).",
        ),
    )

    parser.add_argument(
        "--n-chains",
        type=int,
        default=4,
        help="Number of MCMC chains to run (default: 4).",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help=(
            "Number of posterior samples to draw per chain using NUTS (default: 1000)."
        ),
    )

    parser.add_argument(
        "--exclude-last-n-days",
        type=int,
        default=0,
        help=(
            "Optionally exclude the final n days of available training "
            "data (Default: 0, i.e. exclude no available data"
        ),
    )

    parser.add_argument(
        "--eval-data-path",
        type=Path,
        help=("Path to a parquet file containing compehensive truth data."),
    )

    args = parser.parse_args()
    n_denominator_samples = args.n_samples * args.n_chains
    delattr(args, "n_samples")
    delattr(args, "n_chains")
    main(**vars(args), n_denominator_samples=n_denominator_samples)
