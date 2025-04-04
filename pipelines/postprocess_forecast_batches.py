#!/usr/bin/env python3

"""
Postprocess batches of forecasts, creating
summary files including collated PDFs of plots
and .tsv-format hubverse tables.
"""

import argparse
import datetime
import itertools
import logging
import os
import subprocess
from pathlib import Path

import collate_plots as cp

from pipelines.hubverse_create_observed_data_tables import (
    save_observed_data_tables,
)
from pipelines.utils import get_all_forecast_dirs, parse_model_batch_dir_name


def _hubverse_table_filename(
    report_date: str | datetime.date, disease: str
) -> None:
    return f"{report_date}-{disease.lower()}-hubverse-table.parquet"


def create_hubverse_table(
    model_batch_dir_path: str | Path, locations_exclude: list[str]
) -> None:
    model_batch_dir_path = Path(model_batch_dir_path)
    model_batch_dir_name = model_batch_dir_path.name
    batch_info = parse_model_batch_dir_name(model_batch_dir_name)

    output_file_name = _hubverse_table_filename(
        batch_info["report_date"], batch_info["disease"]
    )

    output_path = Path(model_batch_dir_path, output_file_name)

    result = subprocess.run(
        [
            "Rscript",
            "pipelines/hubverse_create_table.R",
            f"{model_batch_dir_path}",
            f"{output_path}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"create_hubverse_table: {result.stdout}\n{result.stderr}"
        )
    return None


def create_pointinterval_plot(
    model_batch_dir_path: Path | str, locations_exclude: list[str]
) -> None:
    model_batch_dir_path = Path(model_batch_dir_path)
    f_info = parse_model_batch_dir_name(model_batch_dir_path.name)
    disease = f_info["disease"]
    report_date = f_info["report_date"]
    output_file_name = "disease_category_pointintervals.pdf"

    hubverse_table_path = Path(
        model_batch_dir_path, _hubverse_table_filename(report_date, disease)
    )

    figures_dir = Path(model_batch_dir_path, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    output_path = Path(figures_dir, output_file_name)

    result = subprocess.run(
        [
            "Rscript",
            "pipelines/plot_category_pointintervals.R",
            f"{hubverse_table_path}",
            f"{output_path}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"create_pointinterval_plot: {result.stderr}")
    return None


def process_model_batch_dir(
    model_batch_dir_path: Path,
    locations_exclude: list[str],
    plot_ext: str = "pdf",
) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Collating plots...")
    cp.merge_and_save_pdfs(model_batch_dir_path, locations_exclude)
    logger.info("Creating hubverse table...")
    create_hubverse_table(model_batch_dir_path, locations_exclude)
    logger.info("Creating pointinterval plot...")
    create_pointinterval_plot(model_batch_dir_path, locations_exclude)


def main(
    base_forecast_dir: Path | str,
    path_to_latest_data: Path | str,
    diseases: list[str] = ["COVID-19", "Influenza"],
    locations_exclude: list[str] = [
        "AS",
        "GU",
        "MO",
        "MP",
        "PR",
        "UM",
        "VI",
    ],
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    to_process = get_all_forecast_dirs(base_forecast_dir, diseases)
    for batch_dir in to_process:
        logger.info(f"Processing {batch_dir}...")
        model_batch_dir_path = Path(base_forecast_dir, batch_dir)
        process_model_batch_dir(model_batch_dir_path, locations_exclude)
        logger.info(f"Finished processing {batch_dir}")
    logger.info("Created observed data tables for visualization...")
    save_observed_data_tables(
        path_to_latest_data,
        base_forecast_dir,
        daily_filename="daily.tsv",
        epiweekly_filename="epiweekly.tsv",
    )
    logger.info(f"Finished processing {base_forecast_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Postprocess forecasts across locations."
    )
    parser.add_argument(
        "base_forecast_dir",
        type=Path,
        help="Directory containing forecast subdirectories.",
    )
    parser.add_argument(
        "path_to_latest_data",
        type=Path,
        help=("Path to a parquet file containing the latest observed data."),
    )
    parser.add_argument(
        "--diseases",
        type=str,
        default="COVID-19 Influenza",
        help=(
            "Name(s) of disease(s) to postprocess, "
            "as a whitespace-separated string. Supported "
            "values are 'COVID-19' and 'Influenza'. "
            "Default 'COVID-19 Influenza' (i.e. postprocess both)."
        ),
    )
    parser.add_argument(
        "--locations-exclude",
        type=str,
        default="AS GU MO MP PR UM VI",
        help=(
            "Two-letter USPS location abbreviations to "
            "exclude from the job, as a whitespace-separated "
            "string. Defaults to a set of locations for which "
            "we typically do not have available NSSP ED visit "
            "data: 'AS GU MO MP PR UM VI'."
        ),
    )

    args = parser.parse_args()
    args.diseases = args.diseases.split()
    args.locations_exclude = args.locations_exclude.split()
    main(**vars(args))
