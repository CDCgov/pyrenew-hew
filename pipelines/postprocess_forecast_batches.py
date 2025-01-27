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
from forecasttools.utils import ensure_listlike

from pipelines.hubverse_create_observed_data_tables import (
    save_observed_data_tables,
)
from pipelines.utils import get_all_forecast_dirs, parse_model_batch_dir_name


def _hubverse_table_filename(
    report_date: str | datetime.date, disease: str
) -> None:
    return f"{report_date}-" f"{disease.lower()}-" "hubverse-table.tsv"


def create_hubverse_table(
    model_batch_dir_path: str | Path,
    locations_exclude: str | list[str] = "",
    epiweekly_other_locations: str | list[str] = "",
) -> None:
    logger = logging.getLogger(__name__)

    locations_exclude = ensure_listlike(locations_exclude)
    epiweekly_other_locations = ensure_listlike(epiweekly_other_locations)

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
            "--exclude",
            f"{' '.join(locations_exclude)}",
            "--epiweekly-other-locations",
            f"{' '.join(epiweekly_other_locations)}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"create_hubverse_table: {result.stdout}\n {result.stderr}"
        )

    return None


def create_pointinterval_plot(model_batch_dir_path: Path | str) -> None:
    model_batch_dir_path = Path(model_batch_dir_path)
    f_info = parse_model_batch_dir_name(model_batch_dir_path.name)
    disease = f_info["disease"]
    report_date = f_info["report_date"]
    output_file_name = "Disease_category_pointintervals.pdf"

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
            f"{disease}",
            f"{output_path}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("create_pointinterval_plot: " f"{result.stderr}")
    return None


def process_model_batch_dir(
    model_batch_dir_path: Path,
    locations_exclude: str | list[str] = "",
    epiweekly_other_locations: str | list[str] = "",
    plot_ext: str = "pdf",
) -> None:
    locations_exclude = ensure_listlike(locations_exclude)
    epiweekly_other_locations = ensure_listlike(epiweekly_other_locations)

    plot_types = ["Disease", "Other", "prop_disease_ed_visits"]
    plot_timescales = ["daily", "epiweekly", "epiweekly_with_epiweekly_other"]
    plot_yscales = ["", "log_"]

    plots_to_collate = [
        f"{p_type}_forecast_plot_{p_yscale}{p_timescale}.{plot_ext}"
        for p_type, p_yscale, p_timescale in itertools.product(
            plot_types, plot_yscales, plot_timescales
        )
        if not (
            p_type == "Disease"
            and p_timescale == "epiweekly_with_epiweekly_other"
        )
    ]

    logger = logging.getLogger(__name__)
    logger.info("Collating plots...")
    cp.process_dir(model_batch_dir_path, target_filenames=plots_to_collate)
    logger.info("Creating hubverse table...")
    logger.info(
        "Using epiweekly other forecast for " f"{epiweekly_other_locations}..."
    )
    create_hubverse_table(
        model_batch_dir_path,
        locations_exclude=locations_exclude,
        epiweekly_other_locations=epiweekly_other_locations,
    )
    logger.info("Creating pointinterval plot...")
    create_pointinterval_plot(model_batch_dir_path)


def main(
    base_forecast_dir: Path | str,
    path_to_latest_data: Path | str,
    diseases: list[str] = None,
    locations_exclude: str | list[str] = "",
    epiweekly_other_locations: str | list[str] = "",
) -> None:
    if diseases is None:
        diseases = ["COVID-19", "Influenza"]
    diseases = ensure_listlike(diseases)
    locations_exclude = ensure_listlike(locations_exclude)
    epiweekly_other_locations = ensure_listlike(epiweekly_other_locations)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    to_process = get_all_forecast_dirs(base_forecast_dir, diseases)
    for batch_dir in to_process:
        logger.info(f"Processing {batch_dir}...")
        model_batch_dir_path = Path(base_forecast_dir, batch_dir)
        process_model_batch_dir(
            model_batch_dir_path,
            locations_exclude=locations_exclude,
            epiweekly_other_locations=epiweekly_other_locations,
        )
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
        default="",
        help=(
            "Name(s) of locations to exclude from the hubverse table, "
            "as a whitespace-separated string of two-letter location "
            "codes."
        ),
    )
    parser.add_argument(
        "--epiweekly-other-locations",
        type=str,
        default="",
        help=(
            "Name(s) of locations for which to use an explicitly epiweekly "
            "forecast of other (non-target) ED visits, as opposed to a "
            "daily forecast aggregated to epiweekly. Locations should be "
            "specified as a whitespace-separated string of two-letter codes."
        ),
    )

    args = parser.parse_args()
    args.diseases = args.diseases.split()
    args.locations_exclude = args.locations_exclude.split()
    args.epiweekly_other_locations = args.epiweekly_other_locations.split()
    main(**vars(args))
