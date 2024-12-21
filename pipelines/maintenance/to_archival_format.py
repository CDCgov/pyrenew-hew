"""
Transform a forecast archives to the current
archival format
"""

import logging
import os
import shutil
from pathlib import Path

import polars as pl

from pipelines.utils import get_all_forecast_dirs, get_all_model_run_dirs

logging.basicConfig(level=None)


def _csv_to_tsv(path: Path):
    logger = logging.getLogger(__name__)
    path = Path(path)
    if not path.suffix == ".csv":
        raise ValueError("File to convert must have extension .csv")
    outpath = path.with_suffix(".tsv")
    logger.info(f"Converting {path} from csv to tsv...")
    pl.read_csv(path).write_csv(outpath, separator="\t")
    logger.info("Done.")


def convert_files(model_run_subdir_path: Path, dry_run: bool = True):
    csv_to_tsv = ["data", "epiweekly_data"]
    filepaths = [Path(f) for f in os.scandir(model_run_subdir_path)]
    to_convert_csv = [
        fp
        for fp in filepaths
        if fp.stem in csv_to_tsv
        and fp.suffix == ".csv"
        and not fp.with_suffix(".tsv") in filepaths
    ]
    [_csv_to_tsv(fp) for fp in to_convert_csv]


def transform_model_run_dir(
    model_run_dir_path: Path, dry_run: bool = True
) -> None:
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

    dir_contents = [x for x in os.scandir(model_run_dir_path)]

    target_files = dict(
        pyrenew_e=[
            "posterior_samples.pickle",
            "inference_data.nc",
            "inference_data.csv",
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
        to_move = [f for f in dir_contents if f.name in target_files]
        target_dir_path = Path(model_run_dir_path, target_dir)
        if dry_run:
            logger.info(
                "Dry run. A non-dry run would move "
                f"files {to_move} to {target_dir_path}"
            )
        else:
            os.makedirs(target_dir_path, exist_ok=True)
            [shutil.move(x, target_dir_path) for x in to_move]
            convert_files(target_dir_path)


to_transform = [
    Path(
        "/home/dylan/blobfuse/mounts/pyrenew-hew-prod-output", x + "_forecasts"
    )
    for x in ["2024-11-13"]
]


for t_dir in to_transform:
    f_dirs = get_all_forecast_dirs(t_dir, ["COVID-19", "Influenza"])
    for f_dir in f_dirs:
        f_dir_path = Path(t_dir, f_dir)
        mr_dirs = get_all_model_run_dirs(f_dir_path)
        for mr_dir in mr_dirs:
            mr_dir_path = Path(f_dir_path, mr_dir)
            transform_model_run_dir(mr_dir_path, dry_run=False)
