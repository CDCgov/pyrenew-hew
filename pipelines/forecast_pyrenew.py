import argparse
import datetime as dt
import logging
import os
import subprocess
import tomllib
from pathlib import Path

import tomli_w
from fit_pyrenew_model import fit_and_save_model
from generate_predictive import (
    generate_and_save_predictions,
)
from prep_data import get_training_dates_and_model_dir
from pygit2.repository import Repository

from pyrenew_hew.utils import (
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
    loc: str,
    model_run_dir: Path | str,
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
        f"model {pyrenew_model_name}, location {loc}."
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

    timeseries_model_name = "ts_ensemble_e" if fit_ed_visits else None

    if fit_ed_visits and not os.path.exists(
        Path(model_run_dir, timeseries_model_name)
    ):
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
        f"location {loc}."
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create fit data for disease modeling."
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
        "--model-run-dir",
        type=Path,
        required=True,
        help="Directory in which to save output.",
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
