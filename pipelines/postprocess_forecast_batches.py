#!/usr/bin/env python3

"""
Postprocess batches of forecasts, creating
summary files including collated PDFs of plots
and .tsv-format hubverse tables.
"""

import argparse
import datetime as dt
import logging
from pathlib import Path

import collate_plots as cp

from pipelines.common_utils import run_r_script
from pipelines.utils import get_all_forecast_dirs, parse_model_batch_dir_name


def _hubverse_table_filename(report_date: str | dt.date, disease: str) -> None:
    return f"{report_date}-{disease.lower()}-hubverse-table.parquet"


def combine_hubverse_tables(model_batch_dir_path: str | Path) -> None:
    model_batch_dir_path = Path(model_batch_dir_path)
    model_batch_dir_name = model_batch_dir_path.name
    batch_info = parse_model_batch_dir_name(model_batch_dir_name)

    output_file_name = _hubverse_table_filename(
        batch_info["report_date"], batch_info["disease"]
    )

    output_path = Path(model_batch_dir_path, output_file_name)

    run_r_script(
        "pipelines/combine_hubverse_tables.R",
        [
            f"{model_batch_dir_path}",
            f"{output_path}",
        ],
        function_name="combine_hubverse_tables",
    )
    return None


def process_model_batch_dir(model_batch_dir_path: Path, plot_ext: str = "pdf") -> None:
    logger = logging.getLogger(__name__)
    logger.info("Collating plots...")
    cp.merge_and_save_pdfs(model_batch_dir_path)
    logger.info("Creating hubverse table...")
    combine_hubverse_tables(model_batch_dir_path)


def main(
    base_forecast_dir: Path | str,
    diseases: list[str] = ["COVID-19", "Influenza", "RSV"],
    skip_existing: bool = True,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    to_process = get_all_forecast_dirs(base_forecast_dir, list(diseases))
    for batch_dir in to_process:
        model_batch_dir_path = Path(base_forecast_dir, batch_dir)
        hubverse_tbl_exists = bool(
            list(model_batch_dir_path.glob("*-hubverse-table.parquet"))
        )
        if hubverse_tbl_exists and skip_existing:
            logger.info(f"Skipping {batch_dir}, hubverse table already exists.")
        else:
            logger.info(f"Processing {batch_dir}...")
            process_model_batch_dir(model_batch_dir_path)
            logger.info(f"Finished processing {batch_dir}")
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
        "--diseases",
        type=str,
        default="COVID-19 Influenza RSV",
        help=(
            "Name(s) of disease(s) to postprocess, "
            "as a whitespace-separated string. Supported "
            "values are 'COVID-19' , 'RSV' and 'Influenza'. "
            "Default 'COVID-19 Influenza RSV' (i.e. postprocess all)."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing for model batch directories that already have been processed.",
    )

    args = parser.parse_args()
    args.diseases = args.diseases.split()
    main(**vars(args))
