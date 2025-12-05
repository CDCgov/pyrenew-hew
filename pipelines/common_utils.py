"""Common utility functions for forecast pipelines."""

import datetime as dt
import logging
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import polars as pl


def load_credentials(
    credentials_path: Path | str | None, logger: logging.Logger
) -> dict[str, Any] | None:
    """Load credentials from a TOML file."""
    if credentials_path is not None:
        cp = Path(credentials_path)
        if not cp.suffix.lower() == ".toml":
            raise ValueError(
                "Credentials file must have the extension "
                "'.toml' (not case-sensitive). Got "
                f"{cp.suffix}"
            )
        logger.info(f"Reading in credentials from {cp}...")
        with open(cp, "rb") as fp:
            credentials_dict = tomllib.load(fp)
    else:
        logger.info("No credentials file given. Will proceed without one.")
        credentials_dict = None

    return credentials_dict


def get_available_reports(
    data_dir: str | Path, glob_pattern: str = "*.parquet"
) -> list[dt.date]:
    """Get available report dates from glob pattern matching files in a directory. Default pattern matches parquet files."""
    return [
        dt.datetime.strptime(f.stem, "%Y-%m-%d").date()
        for f in Path(data_dir).glob(glob_pattern)
    ]


def parse_and_validate_report_date(
    report_date: str,
    available_facility_level_reports: list[dt.date],
    available_loc_level_reports: list[dt.date],
    logger: logging.Logger,
) -> tuple[dt.date, dt.date | None]:
    """
    Parse and validate report date, determine location-level report date to use.

    Parameters
    ----------
    report_date : str
        Report date as string ("latest" or "YYYY-MM-DD" format).
    available_facility_level_reports : list[dt.date]
        List of available facility-level report dates.
    available_loc_level_reports : list[dt.date]
        List of available location-level report dates.
    logger : logging.Logger
        Process logger.

    Returns
    -------
    tuple[dt.date, dt.date | None]
        Tuple of (report_date, loc_report_date).

    Raises
    ------
    ValueError
        If report date is invalid or data is missing.
    """
    first_available_loc_report = min(available_loc_level_reports)
    last_available_loc_report = max(available_loc_level_reports)

    if report_date == "latest":
        report_date = max(available_facility_level_reports)
    else:
        report_date = dt.datetime.strptime(report_date, "%Y-%m-%d").date()

    if report_date in available_loc_level_reports:
        loc_report_date = report_date
    elif report_date > last_available_loc_report:
        loc_report_date = last_available_loc_report
    elif report_date > first_available_loc_report:
        raise ValueError(
            "Dataset appear to be missing some state-level "
            f"reports. First entry is {first_available_loc_report}, "
            f"last is {last_available_loc_report}, but no entry "
            f"for {report_date}"
        )
    else:
        raise ValueError(
            "Requested report date is earlier than the first "
            "state-level vintage. This is not currently supported"
        )

    logger.info(f"Report date: {report_date}")
    if loc_report_date is not None:
        logger.info(f"Using location-level data as of: {loc_report_date}")

    return report_date, loc_report_date


def calculate_training_dates(
    report_date: dt.date,
    n_training_days: int,
    exclude_last_n_days: int,
    logger: logging.Logger,
) -> tuple[dt.date, dt.date]:
    """
    Calculate first and last training dates.

    Parameters
    ----------
    report_date : dt.date
        The report date.
    n_training_days : int
        Number of training days.
    exclude_last_n_days : int
        Number of days to exclude from the end of training data.
    logger : logging.Logger
        Process logger.

    Returns
    -------
    tuple[dt.date, dt.date]
        Tuple of (first_training_date, last_training_date).

    Raises
    ------
    ValueError
        If last training date is not before report date.
    """
    # + 1 because max date in dataset is report_date - 1
    last_training_date = report_date - dt.timedelta(days=exclude_last_n_days + 1)

    if last_training_date >= report_date:
        raise ValueError(
            "Last training date must be before the report date. "
            f"Got a last training date of {last_training_date} "
            f"with a report date of {report_date}."
        )

    logger.info(f"last training date: {last_training_date}")

    first_training_date = last_training_date - dt.timedelta(days=n_training_days - 1)

    logger.info(f"First training date {first_training_date}")

    return first_training_date, last_training_date


def load_nssp_data(
    report_date: dt.date,
    loc_report_date: dt.date | None,
    available_facility_level_reports: list[dt.date],
    available_loc_level_reports: list[dt.date],
    facility_level_nssp_data_dir: Path,
    state_level_nssp_data_dir: Path,
    logger: logging.Logger,
) -> tuple[pl.LazyFrame | None, pl.LazyFrame | None]:
    """
    Load facility-level and location-level NSSP data.

    Parameters
    ----------
    report_date : dt.date
        The report date.
    loc_report_date : dt.date | None
        The location-level report date to use.
    available_facility_level_reports : list[dt.date]
        List of available facility-level report dates.
    available_loc_level_reports : list[dt.date]
        List of available location-level report dates.
    facility_level_nssp_data_dir : Path
        Directory containing facility-level NSSP data.
    state_level_nssp_data_dir : Path
        Directory containing state-level NSSP data.
    logger : logging.Logger
        Logger for informational messages.

    Returns
    -------
    tuple[pl.LazyFrame | None, pl.LazyFrame | None]
        Tuple of (facility_level_nssp_data, loc_level_nssp_data).

    Raises
    ------
    ValueError
        If no data is available for the requested report date.
    """
    facility_level_nssp_data, loc_level_nssp_data = None, None

    if report_date in available_facility_level_reports:
        logger.info("Facility level data available for the given report date")
        facility_datafile = f"{report_date}.parquet"
        facility_level_nssp_data = pl.scan_parquet(
            Path(facility_level_nssp_data_dir, facility_datafile)
        )
    if loc_report_date in available_loc_level_reports:
        logger.info("location-level data available for the given report date.")
        loc_datafile = f"{loc_report_date}.parquet"
        loc_level_nssp_data = pl.scan_parquet(
            Path(state_level_nssp_data_dir, loc_datafile)
        )
    if facility_level_nssp_data is None and loc_level_nssp_data is None:
        raise ValueError(
            f"No data available for the requested report date {report_date}"
        )

    return facility_level_nssp_data, loc_level_nssp_data


def run_r_script(
    script_name: str,
    args: list[str],
    function_name: str | None = None,
    capture_output: bool = True,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run an R script or command and handle errors.

    Parameters
    ----------
    script_name : str
        Name of the R script to run, or "-e" for inline R code.
    args : list[str]
        Arguments to pass to the R script.
    function_name : str | None, optional
        Name of the calling function for error messages. If None, uses script_name.
    capture_output : bool, optional
        Whether to capture stdout and stderr, by default True.
    text : bool, optional
        Whether to decode output as text, by default False.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process result.

    Raises
    ------
    RuntimeError
        If the R script execution fails.
    """
    command = ["Rscript", script_name] + args

    result = subprocess.run(
        command,
        capture_output=capture_output,
        text=text,
    )

    if result.returncode != 0:
        error_name = function_name or script_name
        error_msg = result.stderr.decode("utf-8") if not text else result.stderr
        raise RuntimeError(f"{error_name}: {error_msg}")

    return result
