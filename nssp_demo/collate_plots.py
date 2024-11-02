#!/usr/bin/env python3

import argparse
import fnmatch
import os
from pathlib import Path

from pypdf import PdfWriter
from utils import get_all_forecast_dirs


def write_merged_pdf(
    to_merge: list[Path | str], output_path: Path | str
) -> None:
    """
    Merge an ordered list of PDF files and
    write the result to a designated output path.

    Parameters
    ----------
    to_merge
        List of paths to the PDF files to merge.

    output_path
        Where to write the merged PDF.
    """
    pdf_writer = PdfWriter()
    for pdf_file in to_merge:
        pdf_writer.append(pdf_file)
    pdf_writer.write(output_path)


def merge_pdfs_from_subdirs(
    base_dir: str | Path,
    file_name: str,
    output_file_name: str = None,
    subdir_pattern="*",
) -> None:
    """
    Find matching PDF files from a set of
    subdirectories of a base directory
    and merge them, writing the resulting
    merged PDF file the base directory.

    Parameters
    ----------
    base_dir
       The base directory in which to save
       the resultant merged PDF file.

    file_name
       Name of the files to merge. Must be an
       exact match.

    output_file_name
       Name for the merged PDF file, which will be
       saved within ``base_dir``. If ``None``,
       use ``file_name``. Default ``None``.

    subdir_pattern
       Unix-shell style wildcard pattern that
       subdirectories must match to be included.
       Default ``'*'`` (match everything).
       See documentation for :func:`fnmatch.fnmatch`
       for details.

    Returns
    -------
    None
    """
    subdirs = [
        f.name
        for f in os.scandir(base_dir)
        if f.is_dir() and fnmatch.fnmatch(f.name, subdir_pattern)
    ]
    to_merge = [
        Path(base_dir, subdir, file_name)
        for subdir in subdirs
        if os.path.exists(Path(base_dir, subdir, file_name))
    ]

    if output_file_name is None:
        output_file_name = file_name

    write_merged_pdf(to_merge, Path(base_dir, output_file_name))

    return None


def main(
    model_base_dir: str | Path, disease: str, target_filenames: list[str]
) -> None:
    """
    Collate target plots for a given disease
    from a given base directory.
    """

    # define a collation function
    def process_dir(dir_name, file_prefix=""):
        for file_name in target_filenames:
            merge_pdfs_from_subdirs(
                dir_name, file_name, output_file_name=file_prefix + file_name
            )

    forecast_dirs = get_all_forecast_dirs(model_base_dir)

    # first collate locations for a given date
    for f_dir in forecast_dirs:
        process_dir(f_dir)

    # then collate dates, adding the disease name
    # as a prefix for disambiguation since the
    # top-level directory may contain forecasts
    # for multiple diseases.
    process_dir(model_base_dir, file_prefix=disease)

    return None


parser = argparse.ArgumentParser(
    description=(
        "Collate forecast plots from subdirectories " "into single PDFs"
    )
)

parser.add_argument(
    "model_base_dir",
    type=Path,
    help=(
        "Base directory containing subdirectories that represent "
        "individual forecast dates, each of which in turn has "
        "subdirectories that represent individual location forecasts"
    ),
)

parser.add_argument(
    "disease", type=str, help="Name of the disease for which to collate plots"
)

parser.add_argument(
    "--target-filenames",
    type=str,
    default=(
        "Disease_forecast_plot.pdf Other_forecast_plot.pdf "
        "prop_disease_ed_visits_forecast_plot.pdf"
    ),
    help=(
        "Exact filenames of PDF files to find and merge, including "
        "the file extension but without the directory path, as "
        "a whitespace-separated string"
    ),
)

if __name__ == "__main__":
    args = parser.parse_args()
    target_filenames = args.target_filenames.split()
    main(
        model_base_dir=args.model_base_dir,
        disease=args.disease,
        target_filenames=target_filenames,
    )
