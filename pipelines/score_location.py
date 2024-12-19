import argparse
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpyro
from save_eval_data import save_eval_data
from utils import parse_model_batch_dir_name

numpyro.set_host_device_count(4)

from fit_model import fit_and_save_model  # noqa
from generate_predictive import generate_and_save_predictions  # noqa


def generate_epiweekly(model_run_dir: Path) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/generate_epiweekly.R",
            f"{model_run_dir}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"generate_epiweekly: {result.stderr}")
    return None


def timeseries_forecasts(
    model_run_dir: Path, model_name: str, n_forecast_days: int, n_samples: int
) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/timeseries_forecasts.R",
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
        raise RuntimeError(f"timeseries_forecasts: {result.stderr}")
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
            f"convert_inferencedata_to_parquet: {result.stderr}"
        )
    return None


def postprocess_forecast(
    model_run_dir: Path, pyrenew_model_name: str, timeseries_model_name: str
) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/postprocess_state_forecast.R",
            f"{model_run_dir}",
            "--pyrenew-model-name",
            f"{pyrenew_model_name}",
            "--timeseries-model-name",
            f"{timeseries_model_name}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"postprocess_forecast: {result.stderr}")
    return None


def score_forecast(model_run_dir: Path) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/score_forecast.R",
            f"{model_run_dir}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"score_forecast: {result.stderr}")
    return None


def render_webpage(model_run_dir: Path) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/render_webpage.R",
            f"{model_run_dir}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"render_webpage: {result.stderr}")
    return None


def get_available_reports(
    data_dir: str | Path, glob_pattern: str = "*.parquet"
):
    return [
        datetime.strptime(f.stem, "%Y-%m-%d").date()
        for f in Path(data_dir).glob(glob_pattern)
    ]


def main(state, model_batch_dir_path: Path, eval_data_path: Path):
    model_batch_dir_path = Path(model_batch_dir_path)
    eval_data_path = Path(eval_data_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    batch_info = parse_model_batch_dir_name(model_batch_dir_path.name)

    logger.info("Getting eval data...")
    save_eval_data(
        state=state,
        latest_comprehensive_path=eval_data_path,
        output_data_dir=Path(model_run_dir, "data", "eval"),
        last_eval_date=(
            batch_info["report_date"] + timedelta(days=n_forecast_days)
        ),
        **batch_info,
    )

    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly(model_run_dir)

    logger.info("Data preparation complete.")

    logger.info("Fitting model")
    fit_and_save_model(
        model_run_dir,
        "pyrenew_e",
        n_warmup=n_warmup,
        n_samples=n_samples,
        n_chains=n_chains,
    )
    logger.info("Model fitting complete")

    logger.info("Performing posterior prediction / forecasting...")

    n_days_past_last_training = n_forecast_days + exclude_last_n_days
    generate_and_save_predictions(
        model_run_dir, "pyrenew_e", n_days_past_last_training
    )

    logger.info(
        "Performing baseline forecasting and non-target pathogen "
        "forecasting..."
    )
    n_denominator_samples = n_samples * n_chains
    timeseries_forecasts(
        model_run_dir,
        "timeseries_e",
        n_days_past_last_training,
        n_denominator_samples,
    )
    logger.info("All forecasting complete.")

    logger.info("Converting inferencedata to parquet...")
    convert_inferencedata_to_parquet(model_run_dir, "pyrenew_e")
    logger.info("Conversion complete.")

    logger.info("Postprocessing forecast...")
    postprocess_forecast(model_run_dir, "pyrenew_e", "timeseries_e")
    logger.info("Postprocessing complete.")

    logger.info("Rendering webpage...")
    render_webpage(model_run_dir)
    logger.info("Rendering complete.")

    if score:
        logger.info("Scoring forecast...")
        score_forecast(model_run_dir)

    logger.info(
        "Single state pipeline complete "
        f"for state {state} with "
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
        "--state",
        type=str,
        required=True,
        help=(
            "Two letter abbreviation for the state to fit"
            "(e.g. 'AK', 'AL', 'AZ', etc.)."
        ),
    )

    args = parser.parse_args()
    numpyro.set_host_device_count(args.n_chains)
    main(**vars(args))
