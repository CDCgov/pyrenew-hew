import argparse
import logging
import subprocess
from pathlib import Path

import collate_plots as cp
from utils import get_all_forecast_dirs, parse_model_batch_dir_name


def create_hubverse_table(model_batch_dir: Path) -> None:
    batch_info = parse_model_batch_dir_name(model_batch_dir)

    output_file_name = (
        f"{batch_info["report_date"]}-"
        f"{batch_info["disease"].lower()}-"
        "hubverse-table.tsv"
    )

    output_path = Path(model_batch_dir, output_file_name)

    result = subprocess.run(
        [
            "Rscript",
            "pipelines/create_hubverse_table.R",
            f"{model_batch_dir}",
            f"{output_path}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"generate_epiweekly: {result.stderr}")
    return None


def process_model_batch_dir(model_batch_dir: Path) -> None:
    cp.process_dir(model_batch_dir)
    create_hubverse_table(model_batch_dir)


def main(base_forecast_dir: Path):
    to_process = get_all_forecast_dirs(base_forecast_dir)
    for batch_dir in to_process:
        process_model_batch_dir(batch_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Postprocess forecasts across locations."
    )
    parser.add_argument(
        "base_forecast_dir",
        type=Path,
        required=True,
        help="Directory containing forecast subdirectories.",
    )

    args = parser.parse_args()
    main(**vars(args))
