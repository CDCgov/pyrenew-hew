#!/usr/bin/env python3

"""
Postprocess batches of forecasts, creating
summary files including collated PDFs of plots
and .tsv-format hubverse tables.
"""

import argparse
import logging
import subprocess
from pathlib import Path

import collate_plots as cp
from utils import get_all_forecast_dirs, parse_model_batch_dir_name


def _hubverse_table_filename(
    report_date: str | datetime.date, disease: str
) -> None:
    return f"{report_date}-" f"{disease.lower()}-" "hubverse-table.tsv"


def create_hubverse_table(model_batch_dir_path: str | Path) -> None:
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
            "pipelines/create_hubverse_table.R",
            f"{model_batch_dir_path}",
            f"{output_path}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"create_hubverse_table: {result.stderr}")
    return None


def create_pointinterval_plot(model_batch_dir_path: Path | str) -> None:
    model_batch_dir_path = Path(model_batch_dir_path)
    f_info = parse_model_batch_dir_name(model_batch_dir_path.name)
    disease = f_info["disease"]
    report_date = f_info["report_date"]
    output_file_name = (
        f"{report_date}_{disease}" "_category_pointintervals.pdf"
    )

    hubverse_table_path = Path(
        model_batch_dir_path, _hubverse_table_filename(report_date, disease)
    )

    output_path = Path(model_batch_dir_path, output_file_name)

    result = subprocess.run(
        [
            "Rscript",
            "pipelines/.R",
            f"{hubverse_table_path}",
            f"{output_path}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("create_pointinterval_plot: " f"{result.stderr}")
    return None


def process_model_batch_dir(model_batch_dir_path: Path) -> None:
    plot_types = ["Disease", "Other", "prop_disease_ed_visits"]
    plots_to_collate = [f"{x}_forecast_plot.pdf" for x in plot_types] + [
        f"{x}_forecast_plot_log.pdf" for x in plot_types
    ]
    logger = logging.getLogger(__name__)
    logger.info("Collating plots...")
    cp.process_dir(model_batch_dir_path, target_filenames=plots_to_collate)
    logger.info("Creating hubverse table...")
    create_hubverse_table(model_batch_dir_path)
    logger.info("Creating pointinterval plot...")
    create_pointinterval_plot(model_batch_dir_path)


def main(
    base_forecast_dir: Path, diseases: list[str] = ["COVID-19", "Influenza"]
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    to_process = get_all_forecast_dirs(base_forecast_dir, diseases)
    for batch_dir in to_process:
        logger.info(f"Processing {batch_dir}...")
        model_batch_dir_path = Path(base_forecast_dir, batch_dir)
        process_model_batch_dir(model_batch_dir_path)


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

    args = parser.parse_args()
    args.diseases = args.diseases.split()
    main(**vars(args))
