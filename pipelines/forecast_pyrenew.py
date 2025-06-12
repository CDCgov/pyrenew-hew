import argparse
import logging
import os
import shutil
import subprocess
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

import numpyro
import polars as pl
import tomli_w
from prep_data import process_and_save_loc
from prep_eval_data import save_eval_data
from pygit2 import Repository

from pyrenew_hew.util import (
    flags_from_hew_letters,
    pyrenew_model_name_from_flags,
)

numpyro.set_host_device_count(4)

from fit_pyrenew_model import fit_and_save_model  # noqa
from generate_predictive import (  # noqa
    generate_and_save_predictions,
)
from prep_ww_data import clean_nwss_data, preprocess_ww_data


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
    except Exception as e:
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


def copy_and_record_priors(priors_path: Path, model_run_dir: Path):
    metadata_file = Path(model_run_dir, "metadata.toml")
    shutil.copyfile(priors_path, Path(model_run_dir, "priors.py"))

    if metadata_file.exists():
        with open(metadata_file, "rb") as file:
            metadata = tomllib.load(file)
    else:
        metadata = {}

    new_metadata = {
        "priors_path": str(priors_path),
    }

    metadata.update(new_metadata)

    with open(metadata_file, "wb") as file:
        tomli_w.dump(metadata, file)


def generate_epiweekly_data(
    model_run_dir: Path, data_names: str = None
) -> None:
    command = [
        "Rscript",
        "pipelines/generate_epiweekly_data.R",
        f"{model_run_dir}",
    ]
    if data_names is not None:
        command.extend(["--data-names", f"{data_names}"])

    result = subprocess.run(
        command,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"generate_epiweekly_data: {result.stderr.decode('utf-8')}"
        )
    return None


def convert_inferencedata_to_parquet(
    model_run_dir: Path, model_name: str
) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/convert_inferencedata_to_parquet.R",
            f"{model_run_dir}",
            "--model-name",
            f"{model_name}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"convert_inferencedata_to_parquet: {result.stderr.decode('utf-8')}"
        )
    return None


def plot_and_save_loc_forecast(
    model_run_dir: Path,
    n_forecast_days: int,
    pyrenew_model_name: str,
    timeseries_model_name: str,
) -> None:
    command = [
        "Rscript",
        "pipelines/plot_and_save_loc_forecast.R",
        f"{model_run_dir}",
        "--n-forecast-days",
        f"{n_forecast_days}",
        "--pyrenew-model-name",
        f"{pyrenew_model_name}",
    ]
    if timeseries_model_name is not None:
        command.extend(
            [
                "--timeseries-model-name",
                f"{timeseries_model_name}",
            ]
        )

    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"plot_and_save_loc_forecast: {result.stderr.decode('utf-8')}"
        )
    return None


def get_available_reports(
    data_dir: str | Path, glob_pattern: str = "*.parquet"
):
    return [
        datetime.strptime(f.stem, "%Y-%m-%d").date()
        for f in Path(data_dir).glob(glob_pattern)
    ]


def main(
    disease: str,
    report_date: str,
    loc: str,
    facility_level_nssp_data_dir: Path | str,
    state_level_nssp_data_dir: Path | str,
    nwss_data_dir: Path | str,
    param_data_dir: Path | str,
    priors_path: Path | str,
    output_dir: Path | str,
    n_training_days: int,
    n_forecast_days: int,
    n_chains: int,
    n_warmup: int,
    n_samples: int,
    nhsn_data_path: Path | str = None,
    exclude_last_n_days: int = 0,
    eval_data_path: Path = None,
    credentials_path: Path = None,
    fit_ed_visits: bool = False,
    fit_hospital_admissions: bool = False,
    fit_wastewater: bool = False,
    forecast_ed_visits: bool = False,
    forecast_hospital_admissions: bool = False,
    forecast_wastewater: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    pyrenew_model_name = pyrenew_model_name_from_flags(
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )

    logger.info(
        "Starting single-location forecasting pipeline for "
        f"model {pyrenew_model_name}, location {loc}, "
        f"and report date {report_date}"
    )
    signals = ["ed_visits", "hospital_admissions", "wastewater"]

    for signal in signals:
        fit = locals().get(f"fit_{signal}", False)
        forecast = locals().get(f"forecast_{signal}", False)
        if fit and not forecast:
            raise ValueError(
                "This pipeline does not currently support "
                "fitting to but not forecasting a signal. "
                f"Asked to fit but not forecast {signal}."
            )
    any_fit = any([locals().get(f"fit_{signal}", False) for signal in signals])
    if not any_fit:
        raise ValueError(
            "pyrenew_null (fitting to no signals) "
            "is not supported by this pipeline"
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

    available_loc_level_reports = get_available_reports(
        state_level_nssp_data_dir
    )
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
            "Got a last training date of {last_training_date} "
            "with a report date of {report_date}."
        )

    logger.info(f"last training date: {last_training_date}")

    first_training_date = last_training_date - timedelta(
        days=n_training_days - 1
    )

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

    nwss_data_disease_map = {
        "COVID-19": "covid",
        "Influenza": "flu",
    }

    def get_available_nwss_reports(
        data_dir: str | Path,
        glob_pattern: str = f"NWSS-ETL-{nwss_data_disease_map[disease]}-",
    ):
        return [
            datetime.strptime(
                f.stem.removeprefix(glob_pattern), "%Y-%m-%d"
            ).date()
            for f in Path(data_dir).glob(f"{glob_pattern}*")
        ]

    if fit_wastewater:
        available_nwss_reports = get_available_nwss_reports(nwss_data_dir)
        if report_date in available_nwss_reports:
            nwss_data_raw = pl.scan_parquet(
                Path(
                    nwss_data_dir,
                    f"NWSS-ETL-{nwss_data_disease_map[disease]}-{report_date}",
                    "bronze.parquet",
                )
            )
            nwss_data_cleaned = clean_nwss_data(nwss_data_raw).filter(
                (pl.col("location") == loc)
                & (pl.col("date") >= first_training_date)
            )
            loc_level_nwss_data = preprocess_ww_data(
                nwss_data_cleaned.collect()
            )
        else:
            raise ValueError(
                "NWSS data not available for the requested report date "
                f"{report_date}"
            )
    else:
        loc_level_nwss_data = None

    param_estimates = pl.scan_parquet(Path(param_data_dir, "prod.parquet"))
    model_batch_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )

    model_batch_dir = Path(output_dir, model_batch_dir_name)

    model_run_dir = Path(model_batch_dir, "model_runs", loc)
    os.makedirs(model_run_dir, exist_ok=True)

    if fit_ed_visits and not os.path.exists(
        Path(model_run_dir, "timeseries_e")
    ):
        raise ValueError(
            "timeseries_e model run not found. "
            "Please ensure that the timeseries forecasts "
            "for the ED visits (E) signal are generated "
            "before fitting Pyrenew models with the E signal. "
            "If running a batch job, set the flag --model-family "
            "'timeseries' to fit timeseries model."
        )

    logger.info("Recording git info...")
    record_git_info(model_run_dir)

    logger.info(f"Copying and recording priors from {priors_path}...")
    copy_and_record_priors(priors_path, model_run_dir)

    logger.info(f"Processing {loc}")
    process_and_save_loc(
        loc_abb=loc,
        disease=disease,
        facility_level_nssp_data=facility_level_nssp_data,
        loc_level_nssp_data=loc_level_nssp_data,
        loc_level_nwss_data=loc_level_nwss_data,
        report_date=report_date,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        param_estimates=param_estimates,
        model_run_dir=model_run_dir,
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
        output_data_dir=Path(model_run_dir, "data"),
        last_eval_date=report_date + timedelta(days=n_forecast_days),
        credentials_dict=credentials_dict,
        nhsn_data_path=nhsn_data_path,
    )
    logger.info("Done getting eval data.")

    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly_data(model_run_dir)

    logger.info("Data preparation complete.")

    logger.info("Fitting model")
    fit_and_save_model(
        model_run_dir,
        pyrenew_model_name,
        n_warmup=n_warmup,
        n_samples=n_samples,
        n_chains=n_chains,
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )
    logger.info("Model fitting complete")

    logger.info("Performing posterior prediction / forecasting...")

    n_days_past_last_training = n_forecast_days + exclude_last_n_days
    generate_and_save_predictions(
        model_run_dir,
        pyrenew_model_name,
        n_days_past_last_training,
        predict_ed_visits=forecast_ed_visits,
        predict_hospital_admissions=forecast_hospital_admissions,
        predict_wastewater=forecast_wastewater,
    )
    logger.info("All forecasting complete.")

    logger.info("Converting inferencedata to parquet...")
    convert_inferencedata_to_parquet(model_run_dir, pyrenew_model_name)
    logger.info("Conversion complete.")

    logger.info("Postprocessing forecast...")
    if fit_ed_visits:
        timeseries_model_name = "timeseries_e"
    else:
        timeseries_model_name = None

    plot_and_save_loc_forecast(
        model_run_dir,
        n_days_past_last_training,
        pyrenew_model_name,
        timeseries_model_name,
    )

    logger.info("Postprocessing complete.")

    logger.info(
        "Single-location pipeline complete "
        f"for model {pyrenew_model_name}, "
        f"location {loc}, and "
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
        help=(
            "Fit the model corresponding to the provided model letters (e.g. 'he', 'e', 'hew')."
        ),
        required=True,
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
        help=(
            "Directory in which to look for facility-level NSSP ED visit data"
        ),
    )

    parser.add_argument(
        "--state-level-nssp-data-dir",
        type=Path,
        default=Path("private_data", "nssp_state_level_gold"),
        help=(
            "Directory in which to look for state-level NSSP ED visit data."
        ),
    )

    parser.add_argument(
        "--nwss-data-dir",
        type=Path,
        default=Path("private_data", "nwss_vintages"),
        help=("Directory in which to look for NWSS data."),
    )

    parser.add_argument(
        "--param-data-dir",
        type=Path,
        default=Path("private_data", "prod_param_estimates"),
        help=(
            "Directory in which to look for parameter estimates"
            "such as delay PMFs."
        ),
        required=True,
    )

    parser.add_argument(
        "--priors-path",
        type=Path,
        help=(
            "Path to an executible python file defining random variables "
            "that require priors as pyrenew RandomVariable objects."
        ),
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
        "--n-chains",
        type=int,
        default=4,
        help="Number of MCMC chains to run (default: 4).",
    )

    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1000,
        help=(
            "Number of warmup iterations per chain for NUTS (default: 1000)."
        ),
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help=(
            "Number of posterior samples to draw per "
            "chain using NUTS (default: 1000)."
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
        "--additional-forecast-letters",
        type=str,
        help=(
            "Forecast the following signals even if they were not fit. "
            "Fit signals are always forecast."
        ),
        default="he",
    )
    parser.add_argument(
        "--nhsn-data-path",
        type=Path,
        help=("Path to local NHSN data (for local testing)"),
        default=None,
    )

    args = parser.parse_args()
    numpyro.set_host_device_count(args.n_chains)
    fit_flags = flags_from_hew_letters(args.model_letters)
    forecast_flags = flags_from_hew_letters(
        args.model_letters + args.additional_forecast_letters,
        flag_prefix="forecast",
    )
    delattr(args, "model_letters")
    delattr(args, "additional_forecast_letters")
    main(**vars(args), **fit_flags, **forecast_flags)
