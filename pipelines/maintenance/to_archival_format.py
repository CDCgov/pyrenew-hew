"""
Transform a forecast archives to the current
archival format
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import polars as pl

from pipelines.utils import get_all_model_run_dirs

logging.basicConfig(level=logging.INFO)


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


def transform_model_batch_dir(
    model_batch_dir_path: Path, dry_run: bool = True
) -> None:
    """
    Transform a flat model batch directory into the new
    more structured model_batch_dir format.

    Parameters
    ----------
    model_batch_dir_path
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
    model_run_dirs_to_move = get_all_model_run_dirs(model_batch_dir_path)
    model_run_dirs_path = Path(model_batch_dir_path, "model_runs")
    figs_path = Path(model_batch_dir_path, "figures")
    figs = [
        quantity + "_forecast_plot" + suffix + ".pdf"
        for quantity in ["Disease", "Other", "prop_disease_ed_visits"]
        for suffix in ["", "_log"]
    ] + ["Disease_category_pointintervals.pdf"]

    figs_to_move = [
        x for x in os.scandir(model_batch_dir_path) if x.name in figs
    ]
    if dry_run:
        logger.info(
            "Dry run. A non-dry run would move "
            f"files {model_run_dirs_to_move} to {model_run_dirs_path}"
        )
        logger.info(
            "Dry run. A non-dry run would move "
            f"files {[x.name for x in figs_to_move]} to {figs_path}"
        )
        [
            transform_model_run_dir(
                Path(model_batch_dir_path, x), dry_run=True
            )
            for x in model_run_dirs_to_move
        ]
    else:
        os.makedirs(model_run_dirs_path, exist_ok=True)
        os.makedirs(figs_path, exist_ok=True)
        figs = [
            shutil.move(Path(model_batch_dir_path, x), figs_path)
            for x in figs_to_move
        ]
        model_run_dirs_to_move = [
            shutil.move(Path(model_batch_dir_path, x), model_run_dirs_path)
            for x in model_run_dirs_to_move
        ]
    model_run_dir_names = get_all_model_run_dirs(model_run_dirs_path)
    [
        transform_model_run_dir(Path(model_run_dirs_path, x), dry_run=dry_run)
        for x in model_run_dir_names
    ]


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
        if dry_run and len(to_move) > 0:
            logger.info(
                "Dry run. A non-dry run would move "
                f"files {[x.name for x in to_move]} "
                f"to {target_dir_path}"
            )
        else:
            os.makedirs(target_dir_path, exist_ok=True)
            [shutil.move(x, target_dir_path) for x in to_move]
            convert_files(target_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Coerce old model batch directories to "
            "latest archival directory schema."
        )
    )

    parser.add_argument(
        "model_batch_dir_path",
        type=Path,
        help="Path to a model batch directory to coerce.",
    )
    parser.add_argument(
        "--dry-run",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help=(
            "If this flag is provided, perform a dry run. This will "
            "reporting what transformations would be performed but "
            "not actually perform them."
        ),
    )

    args = parser.parse_args()
    transform_model_batch_dir(**vars(args))
