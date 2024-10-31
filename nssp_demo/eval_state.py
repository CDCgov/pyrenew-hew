import argparse
import logging
import os
import subprocess
from pathlib import Path

import polars as pl
from prep_data import (
    _disease_map_nssp,
    parse_model_batch_dir_name,
    process_state_level_data,
)


def postprocess_forecast(model_run_dir: Path) -> None:
    subprocess.run(
        [
            "Rscript",
            "nssp_demo/postprocess_state_forecast.R",
            "--model-run-dir",
            f"{model_run_dir}",
        ]
    )
    return None


def main(
    state: str,
    model_batch_dir_name: str,
    latest_comprehensive_path: Path | str,
    output_data_dir: Path | str,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_batch_dir = Path(output_data_dir, model_batch_dir_name)
    model_run_dir = Path(model_batch_dir, state)

    disease, report_date, first_training_date, last_training_date = (
        parse_model_batch_dir_name(model_batch_dir_name)
    )

    state_level_nssp_data = pl.scan_parquet(latest_comprehensive_path)

    state_level_data = (
        process_state_level_data(
            state_level_nssp_data=state_level_nssp_data,
            state_abb=state,
            disease=disease,
            first_training_date=first_training_date,
        )
        .with_columns(
            pl.when(pl.col("date") <= last_training_date)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test"))
            .alias("data_type"),
        )
        .sort(["date", "disease"])
    )

    state_level_data.write_csv(
        Path(model_run_dir, "eval_data.tsv"), separator="\t"
    )

    postprocess_forecast(model_run_dir)
    return None


parser = argparse.ArgumentParser(
    description="Create fit data for disease modeling."
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


parser.add_argument(
    "--state-level-nssp-data-dir",
    type=Path,
    default=Path("private_data", "nssp_state_level_gold"),
    help=("Directory in which to look for state-level NSSP " "ED visit data."),
)

parser.add_argument(
    "--model_batch_dir_name",
    type=Path,
    required=True,
    help="todo",
)

parser.add_argument(
    "--latest_comprehensive_path",
    type=Path,
    default="private_data/nssp-archival-vintages/latest_comprehensive.parquet",
    help="File path to all comprehensive data.",
)

parser.add_argument(
    "--output-data-dir",
    type=Path,
    default="private_data",
    help="Directory in which to save output data.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    numpyro.set_host_device_count(args.n_chains)
    main(**vars(args))
