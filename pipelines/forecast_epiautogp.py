# Key Features:
# Same Structure as PyRenew Pipeline:

# Git info recording
# Data preparation (NSSP, NHSN)
# Evaluation data gathering
# Model execution
# Post-processing (plots, hubverse tables)
# EpiAutoGP-Specific Adaptations:

# Only supports hospital admissions (NHSN data) currently
# Uses convert_to_epiautogp_format() to transform PyRenew JSON to EpiAutoGP JSON
# Calls Julia model via run_epiautogp_model() function
# EpiAutoGP-specific parameters (n_particles, n_mcmc, n_hmc, n_forecast_draws)
# Post-Processing:

# Finds the generated CSV forecast file
# Converts it to parquet for consistency
# Calls plotting function (though the R script plot_epiautogp_forecast.R still needs to be created)
# Command-Line Interface:

# Same arguments as PyRenew pipeline where applicable
# Additional EpiAutoGP-specific parameters
# Example usage:
# uv run python pipelines/forecast_epiautogp.py \
#   --disease COVID-19 \
#   --loc CA \
#   --report-date 2024-12-21 \
#   --param-data-dir private_data/prod_param_estimates \
#   --output-dir private_data \
#   --eval-data-path private_data/eval_data.parquet


import argparse
import datetime as dt
import logging
import os
import subprocess
import tomllib
from pathlib import Path

import polars as pl
import tomli_w
from fit_epiautogp_model import fit_and_save_model
from prep_epiautogp_data import convert_to_epiautogp_format
from pygit2.repository import Repository

from pipelines.prep_data import process_and_save_loc_data, process_and_save_loc_param
from pipelines.prep_eval_data import save_eval_data


def record_git_info(model_run_dir: Path):
    metadata_file = Path(model_run_dir, "metadata.toml")

    if metadata_file.exists():
        with open(metadata_file, "rb") as file:
            metadata = tomllib.load(file)
    else:
        metadata = {}

    try:
        repo = Repository(os.getcwd())
        branch_name = repo.head.shorthand
        commit_sha = str(repo.head.target)
    except Exception:
        branch_name = os.environ.get("GIT_BRANCH_NAME", "unknown")
        commit_sha = os.environ.get("GIT_COMMIT_SHA", "unknown")

    new_metadata = {
        "branch_name": branch_name,
        "commit_sha": commit_sha,
    }

    metadata.update(new_metadata)

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "wb") as file:
        tomli_w.dump(metadata, file)


def get_available_reports(data_dir: str | Path, glob_pattern: str = "*.parquet"):
    return [
        dt.datetime.strptime(f.stem, "%Y-%m-%d").date()
        for f in Path(data_dir).glob(glob_pattern)
    ]


def plot_epiautogp_forecast(
    model_run_dir: Path,
    forecast_csv_path: Path,
) -> None:
    """
    Generate plots for EpiAutoGP forecast using R script.

    Args:
        model_run_dir: Directory containing model run data
        forecast_csv_path: Path to the hubverse-formatted forecast CSV
    """
    # Create figures directory
    figure_dir = Path(model_run_dir, "epiautogp", "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Construct R plotting command
    # Note: This assumes we'll create a simplified R plotting script
    # similar to plot_and_save_loc_forecast.R but for EpiAutoGP
    command = [
        "Rscript",
        "pipelines/plot_epiautogp_forecast.R",
        str(forecast_csv_path),
        str(figure_dir),
        str(model_run_dir),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"plot_epiautogp_forecast failed:\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
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
    nhsn_data_path: Path | str = None,
    exclude_last_n_days: int = 0,
    eval_data_path: Path = None,
    credentials_path: Path = None,
    fit_hospital_admissions: bool = True,
    n_forecast_weeks: int = 4,
    n_particles: int = 500,
    n_mcmc: int = 500,
    n_hmc: int = 250,
    n_forecast_draws: int = 10000,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_name = "epiautogp"

    logger.info(
        "Starting single-location EpiAutoGP forecasting pipeline for "
        f"location {loc} and report date {report_date}"
    )

    # EpiAutoGP currently only supports hospital admissions
    if not fit_hospital_admissions:
        raise ValueError(
            "EpiAutoGP currently only supports hospital admissions forecasting. "
            "fit_hospital_admissions must be True."
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
        report_date = dt.datetime.strptime(report_date, "%Y-%m-%d").date()

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
    last_training_date = report_date - dt.timedelta(days=exclude_last_n_days + 1)

    if last_training_date >= report_date:
        raise ValueError(
            "Last training date must be before the report date. "
            "Got a last training date of {last_training_date} "
            "with a report date of {report_date}."
        )

    logger.info(f"last training date: {last_training_date}")

    first_training_date = last_training_date - dt.timedelta(days=n_training_days - 1)

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

    # EpiAutoGP doesn't currently use parameter estimates, but we keep for consistency
    param_estimates = pl.scan_parquet(Path(param_data_dir, "prod.parquet"))

    model_batch_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )

    model_batch_dir = Path(output_dir, model_batch_dir_name)

    model_run_dir = Path(model_batch_dir, "model_runs", loc)
    os.makedirs(model_run_dir, exist_ok=True)

    logger.info("Recording git info...")
    record_git_info(model_run_dir)

    logger.info(f"Processing {loc}")
    process_and_save_loc_data(
        loc_abb=loc,
        disease=disease,
        facility_level_nssp_data=facility_level_nssp_data,
        loc_level_nssp_data=loc_level_nssp_data,
        loc_level_nwss_data=None,  # EpiAutoGP doesn't use wastewater
        report_date=report_date,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        model_run_dir=model_run_dir,
        logger=logger,
        credentials_dict=credentials_dict,
        nhsn_data_path=nhsn_data_path,
    )

    # Keep for consistency, though EpiAutoGP doesn't use these parameters
    process_and_save_loc_param(
        loc_abb=loc,
        disease=disease,
        loc_level_nwss_data=None,
        param_estimates=param_estimates,
        fit_ed_visits=False,
        model_run_dir=model_run_dir,
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
        last_eval_date=report_date + dt.timedelta(days=n_forecast_days),
        credentials_dict=credentials_dict,
        nhsn_data_path=nhsn_data_path,
    )
    logger.info("Done getting eval data.")

    logger.info("Data preparation complete.")

    # Convert PyRenew data format to EpiAutoGP JSON format
    logger.info("Converting data to EpiAutoGP format...")
    data_json_path = Path(model_run_dir, "data", "data_for_model_fit.json")
    epiautogp_input_path = Path(model_run_dir, "epiautogp_input.json")

    convert_to_epiautogp_format(
        input_json_path=str(data_json_path),
        output_json_path=str(epiautogp_input_path),
        target="nhsn",
        disease=disease,
        location=loc,
        forecast_date=str(last_training_date),
        nowcast_reports_path=None,
    )

    logger.info("Running EpiAutoGP model...")

    fit_and_save_model(
        model_run_dir=model_run_dir,
        model_name=model_name,
        epiautogp_input_json=epiautogp_input_path,
        n_forecast_weeks=n_forecast_weeks,
        n_particles=n_particles,
        n_mcmc=n_mcmc,
        n_hmc=n_hmc,
        n_forecast_draws=n_forecast_draws,
        nthreads=1,
    )

    logger.info("EpiAutoGP model fitting and forecasting complete.")

    # Find the generated forecast CSV file
    epiautogp_output_dir = Path(model_run_dir, model_name)
    forecast_csvs = list(epiautogp_output_dir.glob("*.csv"))
    if len(forecast_csvs) == 0:
        raise RuntimeError(f"No forecast CSV files found in {epiautogp_output_dir}")
    elif len(forecast_csvs) > 1:
        logger.warning(
            f"Multiple CSV files found in {epiautogp_output_dir}. "
            f"Using the first one: {forecast_csvs[0].name}"
        )

    forecast_csv_path = forecast_csvs[0]
    logger.info(f"Found forecast output: {forecast_csv_path.name}")

    logger.info("Postprocessing forecast...")

    # Copy CSV to hubverse_table location for consistency with PyRenew
    hubverse_table_dir = Path(model_run_dir, model_name)
    hubverse_table_path = Path(hubverse_table_dir, "hubverse_table.parquet")

    # Convert CSV to parquet for consistency
    logger.info("Converting hubverse table to parquet format...")
    forecast_df = pl.read_csv(forecast_csv_path)
    forecast_df.write_parquet(hubverse_table_path)

    logger.info("Generating forecast plots...")
    try:
        plot_epiautogp_forecast(model_run_dir, forecast_csv_path)
    except Exception as e:
        logger.warning(f"Plotting failed (continuing anyway): {e}")

    logger.info("Postprocessing complete.")

    logger.info(
        "Single-location EpiAutoGP pipeline complete "
        f"for location {loc} and "
        f"report date {report_date}."
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EpiAutoGP forecasting for a single location."
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
        help=("Directory in which to look for parameter estimates such as delay PMFs."),
        required=True,
    )

    parser.add_argument(
        "--credentials-path",
        type=Path,
        help=("Path to a TOML file containing credentials such as API keys."),
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

    parser.add_argument(
        "--nhsn-data-path",
        type=Path,
        help=("Path to local NHSN data (for local testing)"),
        default=None,
    )

    # EpiAutoGP-specific parameters
    parser.add_argument(
        "--n-forecast-weeks",
        type=int,
        default=4,
        help="Number of weeks to forecast (default: 4).",
    )

    parser.add_argument(
        "--n-particles",
        type=int,
        default=12,
        help="Number of particles for filtering (default: 12).",
    )

    parser.add_argument(
        "--n-mcmc",
        type=int,
        default=100,
        help="Number of MCMC iterations (default: 100).",
    )

    parser.add_argument(
        "--n-hmc",
        type=int,
        default=50,
        help="Number of HMC iterations (default: 50).",
    )

    parser.add_argument(
        "--n-forecast-draws",
        type=int,
        default=2000,
        help="Number of forecast draws (default: 2000).",
    )

    args = parser.parse_args()

    # EpiAutoGP only supports hospital admissions
    main(**vars(args), fit_hospital_admissions=True)
