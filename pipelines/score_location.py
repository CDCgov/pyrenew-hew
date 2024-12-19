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


def main(state, model_batch_dir_path: Path, eval_data_path: Path):
    model_batch_dir_path = Path(model_batch_dir_path)
    eval_data_path = Path(eval_data_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    batch_info = parse_model_batch_dir_name(model_batch_dir_path.name)
    model_run_dir = Path(model_batch_dir_path, "model_runs", state)

    logger.info("Getting eval data...")
    save_eval_data(
        state=state,
        latest_comprehensive_path=eval_data_path,
        output_data_dir=Path(model_run_dir, "data", "eval"),
        last_eval_date=(batch_info["report_date"] + timedelta(days=50)),
        **batch_info,
    )

    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly(model_run_dir)

    logger.info("Scoring forecast...")
    score_forecast(model_run_dir)

    logger.info(f"Scoring complete for state {state}. ")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score a single location's forecast."
    )

    parser.add_argument(
        "model_batch_dir_path",
        type=Path,
        help="Path to a model batch directory of forecasts to score.",
    )

    parser.add_argument(
        "eval_data_path",
        type=Path,
        help=("Path to a parquet file containing evaluation data."),
    )

    parser.add_argument(
        "--state",
        type=str,
        required=True,
        help=(
            "Two letter abbreviation for the state to score"
            "(e.g. 'AK', 'AL', 'AZ', etc.)."
        ),
    )

    args = parser.parse_args()
    main(**vars(args))
