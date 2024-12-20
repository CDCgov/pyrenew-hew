"""
Transform a forecast archives to the current
archival format
"""

import logging
import os
import shutil
from pathlib import Path


def transform_model_run_dir(model_run_dir_path, dry_run=True):
    """
    Transform a flat directory into the new
    model_run_dir format.

    Parameters
    ----------
    model_run_dir_path
        Path to the directory.

    dry_run
        Perform a dry run (report would would be done but do nothing)
        or actually perform the in-place modification?
        Boolean, default ``True`` (perform a dry run).

    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)

    dir_contents = os.scandir(model_run_dir_path)

    target_files = dict(
        pyrenew_e=[
            "posterior_samples.pickle",
            "inference_data.nc",
            "mcmc_tidy",
            "forecast_samples.parquet",
            "forecast_ci.parquet",
            "epiweekly_forecast_samples.parquet",
            "combined_training_eval_data.parquet",
        ],
        timeseries_e=[
            prefix + model + quantity + "_forecast.parquet"
            for prefix in ["", "epiweekly_"]
            for model in ["baseline_cdc_", "baseline_ts_"]
            for quantity in ["count_ed_visits", "prop_ed_visits"]
        ]
        + [
            "other_ed_visits_forecast.parquet",
            "epiweekly_other_ed_visits_forecast.parquet",
        ],
        data=[
            "eval_data.tsv",
            "epiweekly_eval_data.tsv",
            "data.tsv",
            "data.csv",
            "epiweekly_data.tsv",
            "data_for_model_fit.json",
        ],
    )

    for target_dir, target_files in target_files.items():
        to_move = ([f for f in dir_contents if f.name in target_files],)

        if dry_run:
            logger.info(
                "Dry run. A non-dry run would move "
                f"files {to_move} to {target_dir}"
            )
        else:
            os.makedirs(Path(model_run_dir_path, target_dir), exist_ok=True)
            shutil.move(to_move, target_dir)
