import argparse
import subprocess
from pathlib import Path

import collate_plots as cp
from utils import get_all_forecast_dirs, parse_model_batch_dir_name


def create_hubverse_table(base_path: Path, model_batch_dir: Path) -> None:
    batch_info = parse_model_batch_dir_name(model_batch_dir)

    output_file_name = (
        f"{batch_info["report_date"]}-"
        f"{batch_info["disease"].lower()}-"
        "hubverse-table.tsv"
    )

    model_batch_path = Path(base_path, model_batch_dir)
    output_path = Path(model_batch_path, output_file_name)

    result = subprocess.run(
        [
            "Rscript",
            "pipelines/create_hubverse_table.R",
            f"{model_batch_path}",
            f"{output_path}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"generate_epiweekly: {result.stderr}")
    return None


def process_model_batch_dir(base_dir: Path, model_batch_dir: Path) -> None:
    plot_types = ["Disease", "Other", "prop_disease_ed_visits"]
    plots_to_collate = [f"{x}_forecast_plot.pdf" for x in plot_types] + [
        f"{x}_forecast_plot_log.pdf" for x in plot_types
    ]
    cp.process_dir(
        Path(base_dir, model_batch_dir), target_filenames=plots_to_collate
    )
    create_hubverse_table(base_dir, model_batch_dir)


def main(
    base_forecast_dir: Path, diseases: list[str] = ["COVID-19", "Influenza"]
):
    to_process = get_all_forecast_dirs(base_forecast_dir, diseases)
    for batch_dir in to_process:
        process_model_batch_dir(base_forecast_dir, batch_dir)


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
