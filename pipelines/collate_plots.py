#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path

from pypdf import PdfWriter
from utils import ensure_listlike, get_all_forecast_dirs


def merge_pdfs_and_save(
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

    Returns
    -------
    None
    """
    pdf_writer = PdfWriter()
    for pdf_file in to_merge:
        pdf_writer.append(pdf_file)
    pdf_writer.write(output_path)

    return None


def merge_pdfs_from_subdirs(
    base_dir: str | Path,
    file_name: str,
    save_dir: str | Path = None,
    output_file_name: str = None,
    subdirs_only: list[str] = None,
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

    save_dir
       Directory in which to save the merged PDF.
       If ``None``, use a "figures" directory in the parent directory of ``base_dir``.
       Default ``None``.

    output_file_name
       Name for the merged PDF file, which will be
       saved within ``base_dir``. If ``None``,
       use ``file_name``. Default ``None``.

    subdirs_only
       Explicit list of subdirs to process. If
       provided, process only subdirs found
       within the ``base_dir`` that are named
       in this list (and match the ``subdir_pattern``).
       If ``None``, process all subdirs (provided
       they match the ``subdir_pattern``).
       Default ``None``.

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

    if save_dir is None:
        save_dir = Path(base_dir).parent / "figures"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    subdirs = [
        f.name for f in Path(base_dir).glob(subdir_pattern) if f.is_dir()
    ]

    if subdirs_only is not None:
        subdirs = [s for s in subdirs if s in subdirs_only]

    to_merge = [
        Path(base_dir, subdir, file_name)
        for subdir in subdirs
        if os.path.exists(Path(base_dir, subdir, file_name))
    ]

    if output_file_name is None:
        output_file_name = file_name

    if len(to_merge) > 0:
        merge_pdfs_and_save(to_merge, Path(save_dir, output_file_name))

    return None


def process_dir(
    base_dir: Path | str,
    target_filenames: str | list[str],
    save_dir: Path | str = None,
    file_prefix: str = "",
    subdirs_only: list[str] = None,
) -> None:
    """
    Merge groups of PDFs from the subdirectories of
    a given base directory, saving the resulting
    merged PDFs in the base directory.

    Parameters
    ----------
    base_dir
        Path to the base directory in which to look

    target_filenames
        One or more PDFs filenames to look for in the
        subdirectories and merge.

    save_dir
        Directory in which to save the merged PDFs.
        If ``None``, use a "figures" directory in the parent directory of ``base_dir``. Default ``None``.

    file_prefix
        Prefix to append to the names in `target_filenames`
        when naming the merged files.

    subdirs_only
        Only look for files to merge in these specific
        named subdirectories. If ``None``, look in all
        subdirectories of ``base_dir``. Default ``None``.
    """
    if save_dir is None:
        save_dir = Path(base_dir).parent / "figures"

    for file_name in ensure_listlike(target_filenames):
        merge_pdfs_from_subdirs(
            base_dir,
            file_name,
            save_dir,
            output_file_name=file_prefix + file_name,
            subdirs_only=subdirs_only,
        )


def collate_from_all_subdirs(
    model_base_dir: str | Path,
    disease: str,
    target_filenames: str | list[str],
    save_dir: str | Path = None,
) -> None:
    """
    Collate target plots for a given disease
    from a given base directory.

    Parameters
    ----------
    model_base_dir
        Path to the base directory in whose subdirectories
        the script will look for PDFs to merge.

    disease
        Name of the target disease. Merged PDFs will be named
        with the disease as a prefix.

    target_filenames
        One or more PDFs filenames to look for in the
        subdirectories and merge.

    save_dir
        Directory in which to save the merged PDFs.
        If ``None``, use a "figures" directory in the parent directory of ``model_base_dir``. Default ``None``.

    Returns
    -------
    None
    """
    if save_dir is None:
        save_dir = Path(model_base_dir).parent / "figures"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    target_filenames = ensure_listlike(target_filenames)

    forecast_dirs = get_all_forecast_dirs(model_base_dir, diseases=disease)

    # first collate locations for a given date
    logger.info(
        "Collating plots across locations, by forecast date. "
        f"{len(forecast_dirs)} dates to process."
    )
    for f_dir in forecast_dirs:
        logger.info(f"Collating plots from {f_dir}")
        process_dir(
            base_dir=Path(model_base_dir, f_dir),
            target_filenames=target_filenames,
            save_dir=save_dir,
        )
    logger.info("Done collating across locations by date.")

    # then collate dates, adding the disease name
    # as a prefix for disambiguation since the
    # top-level directory may contain forecasts
    # for multiple diseases.
    logger.info("Collating plots from forecast date directories...")
    process_dir(
        base_dir=model_base_dir,
        target_filenames=target_filenames,
        save_dir=save_dir,
        file_prefix=f"{disease}_",
        subdirs_only=forecast_dirs,
    )

    logger.info("Done collating plots from forecast date directories.")

    return None


def main(
    dir_of_forecast_dirs: str | Path,
    single_forecast_dir: str | Path,
    target_filenames: list[str],
    disease: str = None,
) -> None:
    if not ((dir_of_forecast_dirs is None) ^ (single_forecast_dir is None)):
        raise ValueError(
            "Must provide exactly one of "
            "'dir_of_forecast_dirs' (to process multiple "
            "groups of forecasts) or "
            "'single_forecast_dir' "
            "(to process a single set of forecasts"
        )
    elif dir_of_forecast_dirs is not None:
        if disease is None:
            raise ValueError(
                "'disease' must not be None when collating plots "
                "from multiple forecast subdirectories"
            )
        collate_from_all_subdirs(
            dir_of_forecast_dirs, disease, target_filenames
        )
    elif single_forecast_dir is not None:
        process_dir(single_forecast_dir, target_filenames)
    return None


parser = argparse.ArgumentParser(
    description=("Collate forecast plots from subdirectories into single PDFs")
)

parser.add_argument(
    "--dir-of-forecast-dirs",
    type=Path,
    help=(
        "Base directory containing subdirectories that represent "
        "individual forecast dates, each of which in turn has "
        "subdirectories that represent individual location forecasts."
    ),
    default=None,
)

parser.add_argument(
    "--single-forecast-dir",
    type=Path,
    help="Path to a single directory to process",
    default=None,
)

parser.add_argument(
    "--disease",
    type=str,
    help="Name of the disease for which to collate plots.",
    default=None,
)

parser.add_argument(
    "--target-filenames",
    type=str,
    default=(
        "Disease_forecast_plot.pdf Other_forecast_plot.pdf "
        "prop_disease_ed_visits_forecast_plot.pdf "
        "Disease_forecast_plot_log.pdf Other_forecast_plot_log.pdf "
        "prop_disease_ed_visits_forecast_plot_log.pdf"
    ),
    help=(
        "Exact filenames of PDF files to find and merge, including "
        "the file extension but without the directory path, as "
        "a whitespace-separated string"
    ),
)

if __name__ == "__main__":
    args = parser.parse_args()
    args.target_filenames = args.target_filenames.split()
    main(**vars(args))
