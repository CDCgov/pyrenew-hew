#!/usr/bin/env python3

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from pypdf import PdfWriter


def merge_pdfs(to_merge: list[Path], output_path: Path) -> None:
    """
    Merge an ordered list of PDF files and
    write the result to a designated output path.

    Parameters
    ----------
    to_merge
        List of paths to the PDF files to merge.

    output_path
        Where to write the merged PDF.

    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Merging {len(to_merge)} PDFs into {output_path}")
    pdf_writer = PdfWriter()
    for pdf_file in to_merge:
        pdf_writer.append(str(pdf_file))  # pypdf expects a string path
    pdf_writer.write(str(output_path))


def collect_pdfs(model_batch_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Find and group PDFs by their shared filenames within each model folder."""
    model_runs_dir = model_batch_dir / "model_runs"

    if not model_runs_dir.exists():
        print("Error: model_runs directory not found.")
        return {}

    pdf_groups = defaultdict(lambda: defaultdict(list))

    for location_path in model_runs_dir.iterdir():
        if not location_path.is_dir():
            continue

        for model_path in location_path.iterdir():
            if not model_path.is_dir():
                continue

            for file_path in model_path.glob("*.pdf"):
                base_name = file_path.name  # Keep full filename
                pdf_groups[model_path.name][base_name].append(file_path)

    return pdf_groups


def merge_and_save_pdfs(model_batch_dir: Path) -> None:
    """Merge PDFs by shared name and save to model_batch_dir."""
    pdf_groups = collect_pdfs(model_batch_dir)

    if not pdf_groups:
        print("No PDFs found to merge.")
        return

    for model, file_dict in pdf_groups.items():
        for original_file_name, pdf_list in file_dict.items():
            if len(pdf_list) > 1:
                output_filename = f"{model}-{original_file_name}"
                output_path = model_batch_dir / "figures" / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                merge_pdfs(pdf_list, output_path)
            else:
                print(
                    f"Skipping {original_file_name} in {model}, only one file found."
                )


parser = argparse.ArgumentParser(
    description=("Collate forecast plots from subdirectories into single PDFs")
)

parser.add_argument(
    "modelbatchdir",
    type=Path,
    help=(
        "Base directory containing subdirectories that represent "
        "individual forecast dates, each of which in turn has "
        "subdirectories that represent individual location forecasts."
    ),
    default=None,
)

if __name__ == "__main__":
    args = parser.parse_args()
    model_batch_dir = args.modelbatchdir
    merge_and_save_pdfs(model_batch_dir)
