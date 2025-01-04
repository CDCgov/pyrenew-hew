#!/usr/bin/env python3

"""
Score pyrenew-hew forecast output for a single
location against evaluation data.
"""

import argparse
import logging
import subprocess
from datetime import timedelta
from pathlib import Path

from prep_eval_data import save_eval_data
from utils import parse_model_batch_dir_name


def generate_epiweekly(model_run_dir: Path) -> None:
    """
    Run pipelines/generate_epiweekly.R on a given
    model run directory, capturing output.

    Parameters
    ----------
    model_run_dir
        Target model run directory.

    Returns
    -------
    None
    """
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
    """
    Run pipelines/score_forecast.R on the given
    model run directory, capturing output.

    Parameters
    ----------
    model_run_dir
        Target model run directory.

    Returns
    -------
    None
    """
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


def main(
    location: str, model_batch_dir_path: Path, eval_data_path: Path
) -> None:
    """
    Score a pyrenew-hew model run for a given location.

    Parameters
    ----------
    location
        Location to score, as a two-letter location code
        (e.g. "AK", "US").

    model_batch_dir_path
        Model batch directory containing location-specific
        runs to score.

    eval_data_path
        Path to a parquet file containing evaluation data
        against which to score.

    Returns
    -------
    None
    """
    model_batch_dir_path = Path(model_batch_dir_path)
    eval_data_path = Path(eval_data_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    batch_info = parse_model_batch_dir_name(model_batch_dir_path.name)
    model_run_dir_path = Path(model_batch_dir_path, "model_runs", location)

    logger.info("Getting eval data...")
    save_eval_data(
        state=location,
        latest_comprehensive_path=eval_data_path,
        output_data_dir=Path(model_run_dir_path, "data"),
        last_eval_date=(batch_info["report_date"] + timedelta(days=50)),
        **batch_info,
    )

    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly(model_run_dir_path)

    logger.info("Scoring forecast...")
    score_forecast(model_run_dir_path)

    logger.info(f"Scoring complete for location {location}. ")
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
        "--location",
        type=str,
        required=True,
        help=(
            "Two letter abbreviation for the location to score"
            "(e.g. 'AK', 'AL', 'AZ', etc.)."
        ),
    )

    args = parser.parse_args()
    main(**vars(args))
