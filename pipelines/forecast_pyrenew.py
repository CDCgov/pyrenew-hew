import argparse
import datetime as dt
import logging
import os
import shutil
import subprocess
import tomllib
import datetime as dt
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import polars as pl
import tomli_w
from fit_pyrenew_model import fit_and_save_model
from generate_predictive import (
    generate_and_save_predictions,
)
from pygit2.repository import Repository
from prep_data import get_training_dates
from pyrenew_hew.utils import (
    approx_lognorm,
    flags_from_hew_letters,
    pyrenew_model_name_from_flags,
)


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


def convert_inferencedata_to_parquet(model_run_dir: Path, model_name: str) -> None:
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


def _validate_and_extract(
    df: pl.DataFrame,
    parameter_name: str,
    allow_missing_right_truncation: bool = False,
) -> list:
    if (
        allow_missing_right_truncation
        and parameter_name == "right_truncation"
        and df.height == 0
    ):
        return list([1])
    if df.height != 1:
        error_msg = f"Expected exactly one {parameter_name} parameter row, but found {df.height}"
        logging.error(error_msg)
        if df.height > 0:
            logging.error(f"Found rows: {df}")
        raise ValueError(error_msg)
    return df.item(0, "value").to_list()


def get_pmfs(
    param_estimates: pl.LazyFrame,
    loc_abb: str,
    disease: str,
    as_of: dt.date = None,
    reference_date: dt.date = None,
    allow_missing_right_truncation: bool = False,
) -> tuple[list, list, list]:
    """
    Filter and extract probability mass functions (PMFs) from
    parameter estimates LazyFrame based on location, disease
    and date filters.

    This function queries a LazyFrame containing epidemiological
    parameters and returns three types of PMF parameters:
    delay, generation interval, and right truncation.

    Parameters
    ----------
    param_estimates: pl.LazyFrame
        A LazyFrame containing parameter data with columns
        including 'disease', 'parameter', 'value', 'geo_value',
        'start_date', 'end_date', and 'reference_date'.

    loc_abb : str
        Location abbreviation (geo_value) to filter
        right truncation parameters.

    disease : str
        Name of the disease.

    as_of : datetime.date, optional
        Date for which parameters must be valid
        (start_date <= as_of <= end_date). Defaults
        to the most recent estimates.

    reference_date : datetime.date, optional
        The reference date for right truncation estimates.
        Defaults to as_of value. Selects the most recent estimate
        with reference_date <= this value.

    allow_missing_right_truncation : bool, optional
        If true, allows extraction of other pmfs if
        right_truncation estimate is missing

    Returns
    -------
    tuple[list, list, list]
        A tuple containing three arrays:
        - generation_interval_pmf: Generation interval distribution
        - delay_pmf: Delay distribution
        - right_truncation_pmf: Right truncation distribution

    Raises
    ------
    ValueError
        If exactly one row is not found for any of the required parameters.

    Notes
    -----
    The function applies specific filtering logic for each parameter type:
    - For delay and generation_interval: filters by disease,
      parameter name, and validity date range.
    - For right_truncation: additionally filters by location.
    """
    min_as_of = dt.date(1000, 1, 1)
    max_as_of = dt.date(3000, 1, 1)
    as_of = as_of or max_as_of
    reference_date = reference_date or as_of

    filtered_estimates = (
        param_estimates.with_columns(
            pl.col("start_date").fill_null(min_as_of),
            pl.col("end_date").fill_null(max_as_of),
        )
        .filter(pl.col("disease") == disease)
        .filter(
            pl.col("start_date") <= as_of,
            pl.col("end_date") >= as_of,
        )
    )

    generation_interval_df = filtered_estimates.filter(
        pl.col("parameter") == "generation_interval"
    ).collect()

    generation_interval_pmf = _validate_and_extract(
        generation_interval_df, "generation_interval"
    )

    delay_df = filtered_estimates.filter(pl.col("parameter") == "delay").collect()
    delay_pmf = _validate_and_extract(delay_df, "delay")

    # ensure 0 first entry; we do not model the possibility
    # of a zero infection-to-recorded-admission delay in Pyrenew-HEW
    delay_pmf[0] = 0.0
    delay_pmf = jnp.array(delay_pmf)
    delay_pmf = delay_pmf / delay_pmf.sum()
    delay_pmf = delay_pmf.tolist()

    right_truncation_df = (
        filtered_estimates.filter(pl.col("geo_value") == loc_abb)
        .filter(pl.col("parameter") == "right_truncation")
        .filter(pl.col("reference_date") == pl.col("reference_date").max())
        .collect()
    )
    right_truncation_pmf = _validate_and_extract(
        right_truncation_df, "right_truncation", allow_missing_right_truncation
    )

    return (generation_interval_pmf, delay_pmf, right_truncation_pmf)


def main(
    disease: str,
    loc: str,
    report_date: dt.date,
    param_data_dir: Path | str,
    priors_path: Path | str,
    output_dir: Path | str,
    n_training_days: int,
    n_forecast_days: int,
    n_chains: int,
    n_warmup: int,
    n_samples: int,
    exclude_last_n_days: int = 0,
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
            "pyrenew_null (fitting to no signals) " "is not supported by this pipeline"
        )

    param_estimates = pl.scan_parquet(Path(param_data_dir, "prod.parquet"))

    report_date = datetime.date.today()
    logger.info(f"Report date: {report_date}")
    (last_training_date, first_training_date) = get_training_dates(
        report_date, exclude_last_n_days, n_training_days
    )
    model_batch_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )

    model_batch_dir = Path(output_dir, model_batch_dir_name)

    model_run_dir = Path(model_batch_dir, "model_runs", loc)
    os.makedirs(model_run_dir, exist_ok=True)

    timeseries_model_name = "ts_ensemble_e" if fit_ed_visits else None

    if fit_ed_visits and not os.path.exists(Path(model_run_dir, timeseries_model_name)):
        raise ValueError(
            f"{timeseries_model_name} model run not found. "
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
        generation_interval_pmf=generation_interval_pmf,
        right_truncation_pmf=right_truncation_pmf,
        inf_to_hosp_admit_lognormal_loc=inf_to_hosp_admit_lognormal_loc,
        inf_to_hosp_admit_lognormal_scale=inf_to_hosp_admit_lognormal_scale,
        inf_to_hosp_admit_pmf=inf_to_hosp_admit_pmf,
    )
    logger.info("Model fitting complete")

    logger.info("Performing posterior prediction / forecasting...")

    n_days_past_last_training = n_forecast_days + exclude_last_n_days
    generate_and_save_predictions(
        model_run_dir,
        pyrenew_model_name,
        n_days_past_last_training,
        generation_interval_pmf=generation_interval_pmf,
        right_truncation_pmf=right_truncation_pmf,
        inf_to_hosp_admit_lognormal_loc=inf_to_hosp_admit_lognormal_loc,
        inf_to_hosp_admit_lognormal_scale=inf_to_hosp_admit_lognormal_scale,
        inf_to_hosp_admit_pmf=inf_to_hosp_admit_pmf,
        predict_ed_visits=forecast_ed_visits,
        predict_hospital_admissions=forecast_hospital_admissions,
        predict_wastewater=forecast_wastewater,
    )
    logger.info("All forecasting complete.")

    logger.info("Converting inferencedata to parquet...")
    convert_inferencedata_to_parquet(model_run_dir, pyrenew_model_name)
    logger.info("Conversion complete.")

    logger.info("Postprocessing forecast...")

    plot_and_save_loc_forecast(
        model_run_dir,
        n_days_past_last_training,
        pyrenew_model_name,
        timeseries_model_name,
    )

    create_hubverse_table(Path(model_run_dir, pyrenew_model_name))

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
        "--report-date",
        type=dt.date,
        default=dt.date.today(),
        help="Report date in YYYY-MM-DD format",
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
        "--param-data-dir",
        type=Path,
        default=Path("private_data", "prod_param_estimates"),
        help=(
            "Directory in which to look for parameter estimates" "such as delay PMFs."
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
        help=("Number of warmup iterations per chain for NUTS (default: 1000)."),
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
        "--additional-forecast-letters",
        type=str,
        help=(
            "Forecast the following signals even if they were not fit. "
            "Fit signals are always forecast."
        ),
        default="he",
    )

    args = parser.parse_args()
    fit_flags = flags_from_hew_letters(args.model_letters)
    forecast_flags = flags_from_hew_letters(
        args.model_letters + args.additional_forecast_letters,
        flag_prefix="forecast",
    )
    delattr(args, "model_letters")
    delattr(args, "additional_forecast_letters")
    main(**vars(args), **fit_flags, **forecast_flags)
