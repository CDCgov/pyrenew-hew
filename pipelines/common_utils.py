"""Common utility functions for forecast pipelines."""

import datetime as dt
import logging
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import polars as pl

from pipelines.cli_utils import run_command


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


def parse_exclude_date_ranges(
    exclude_date_ranges_str: str | None,
) -> list[tuple[dt.date, dt.date]] | None:
    """
    Parse comma-separated date ranges from string to list of tuples.

    This utility function is useful for parsing date exclusion parameters
    that may be used by various forecasting models to exclude periods with
    known reporting problems or other data quality issues.

    Parameters
    ----------
    exclude_date_ranges_str : str | None
        Comma-separated list of date ranges in format 'start:end'.
        Example: '2024-01-15:2024-01-20,2024-03-01:2024-03-07'

    Returns
    -------
    list[tuple[dt.date, dt.date]] | None
        List of (start_date, end_date) tuples where both dates are inclusive,
        or None if input is None/empty.

    Raises
    ------
    ValueError
        If date range format is invalid, dates can't be parsed as YYYY-MM-DD,
        or start_date > end_date.

    Examples
    --------
    >>> parse_exclude_date_ranges("2024-01-15:2024-01-20")
    [(datetime.date(2024, 1, 15), datetime.date(2024, 1, 20))]

    >>> parse_exclude_date_ranges("2024-01-15:2024-01-20,2024-03-01:2024-03-07")
    [(datetime.date(2024, 1, 15), datetime.date(2024, 1, 20)),
     (datetime.date(2024, 3, 1), datetime.date(2024, 3, 7))]

    >>> parse_exclude_date_ranges(None)
    None
    """
    if exclude_date_ranges_str is None or not exclude_date_ranges_str.strip():
        return None

    parsed_ranges = []
    for date_range_str in exclude_date_ranges_str.split(","):
        date_range_str = date_range_str.strip()
        if ":" not in date_range_str:
            raise ValueError(
                f"Invalid date range format: '{date_range_str}'. "
                "Expected format: 'start_date:end_date' (e.g., '2024-01-15:2024-01-20')"
            )
        start_str, end_str = date_range_str.split(":", 1)
        try:
            start_date = dt.datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
            end_date = dt.datetime.strptime(end_str.strip(), "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(
                f"Invalid date format in range '{date_range_str}'. "
                f"Expected YYYY-MM-DD format. Error: {e}"
            ) from e
        if start_date > end_date:
            raise ValueError(
                f"Invalid date range '{date_range_str}': "
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            )
        parsed_ranges.append((start_date, end_date))

    return parsed_ranges


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
    executor_flags: list[str] | None = None,
    function_name: str | None = None,
    capture_output: bool = True,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run an R script or command and handle errors.

    This is a convenience wrapper around `run_command` for R scripts.
    Supports the pattern: Rscript {FLAGS} {SCRIPT} {ARGS}

    Parameters
    ----------
    script_name : str
        Name of the R script to run.
    args : list[str]
        Arguments to pass to the R script.
    executor_flags : list[str] | None
        Flags to pass to the Rscript executable before the script name.
        For example: ["--vanilla", "--verbose"]
    function_name : str | None
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

    Examples
    --------
    Run a script with vanilla mode:
        >>> run_r_script("script.R", ["--arg1", "value"], executor_flags=["--vanilla"])
    """
    command_args = (executor_flags or []) + [script_name] + args
    return run_command(
        "Rscript",
        command_args,
        function_name=function_name,
        capture_output=capture_output,
        text=text,
    )


def run_r_code(
    r_code: str,
    executor_flags: list[str] | None = None,
    function_name: str | None = None,
    capture_output: bool = True,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run inline R code and handle errors.

    This is a convenience wrapper around `run_r_script` for inline R code.
    Supports the pattern: Rscript {FLAGS} -e {CODE}

    Parameters
    ----------
    r_code : str
        The R code to execute.
    executor_flags : list[str] | None
        Flags to pass to the Rscript executable.
        For example: ["--vanilla", "--verbose"]
    function_name : str | None
        Name of the calling function for error messages.
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
        If the R code execution fails.

    Examples
    --------
    Run R code with vanilla mode:
        >>> run_r_code("print('hello')", executor_flags=["--vanilla"])
    """
    flags_with_inline = (executor_flags or []) + ["-e"]
    return run_r_script(
        r_code,
        [],
        executor_flags=flags_with_inline,
        function_name=function_name,
        capture_output=capture_output,
        text=text,
    )


def run_julia_script(
    script_name: str,
    args: list[str],
    executor_flags: list[str] | None = None,
    function_name: str | None = None,
    capture_output: bool = True,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a Julia script and handle errors.

    This is a convenience wrapper around `run_command` for Julia scripts.
    Supports the pattern:

    > julia {FLAGS} {SCRIPT} {ARGS}

    Parameters
    ----------
    script_name : str
        Name of the Julia script to run.
    args : list[str]
        Arguments to pass to the Julia script.
    executor_flags : list[str] | None
        Flags to pass to the julia executable before the script name.
        Common flags include:
        - ["--project=PATH"] or ["--project=@."] to specify the project environment
        - ["--threads=N"] or ["--threads=auto"] to set number of threads
        - ["--optimize=2"] to set optimization level
        - ["-O3", "--check-bounds=no"] for maximum performance
    function_name : str | None
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
        If the Julia script execution fails.
    """
    command_args = (
        (executor_flags or []) + [script_name] + args
    )  # use "truthy" to handle None
    return run_command(
        "julia",
        command_args,
        function_name=function_name,
        capture_output=capture_output,
        text=text,
    )


def run_julia_code(
    julia_code: str,
    executor_flags: list[str] | None = None,
    function_name: str | None = None,
    capture_output: bool = True,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run inline Julia code and handle errors.

    This is a convenience wrapper around `run_julia_script` for inline Julia code.
    Supports the pattern: julia {FLAGS} -e {CODE}

    Parameters
    ----------
    julia_code : str
        The Julia code to execute.
    executor_flags : list[str] | None
        Flags to pass to the julia executable.
        Common flags include:
        - ["--project=@."] to specify the project environment
        - ["--threads=N"] or ["--threads=auto"] to set number of threads
        - ["--optimize=2"] to set optimization level
    function_name : str | None
        Name of the calling function for error messages.
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
        If the Julia code execution fails.
    """
    flags_with_inline = (executor_flags or []) + ["-e"]
    return run_julia_script(
        julia_code,
        [],
        executor_flags=flags_with_inline,
        function_name=function_name,
        capture_output=capture_output,
        text=text,
    )


def plot_and_save_loc_forecast(
    model_run_dir: Path,
    n_forecast_days: int,
    pyrenew_model_name: str = None,
    timeseries_model_name: str = None,
    model_name: str = None,
) -> None:
    """Plot and save location forecast using R script.

    Parameters
    ----------
    model_run_dir : Path
        Directory containing the model run.
    n_forecast_days : int
        Number of days to forecast.
    pyrenew_model_name : str, optional
        Name of the PyRenew model (legacy).
    timeseries_model_name : str, optional
        Name of the timeseries model (legacy).
    model_name : str, optional
        Generic model name. When provided, auto-detects model type
        and dispatches to appropriate processing method.

    Returns
    -------
    None
    """
    args = [
        f"{model_run_dir}",
        "--n-forecast-days",
        f"{n_forecast_days}",
    ]
    if pyrenew_model_name is not None:
        args.extend(
            [
                "--pyrenew-model-name",
                f"{pyrenew_model_name}",
            ]
        )
    if timeseries_model_name is not None:
        args.extend(
            [
                "--timeseries-model-name",
                f"{timeseries_model_name}",
            ]
        )
    if model_name is not None:
        args.extend(
            [
                "--model-name",
                f"{model_name}",
            ]
        )

    run_r_script(
        "pipelines/plot_and_save_loc_forecast.R",
        args,
        function_name="plot_and_save_loc_forecast",
    )
    return None


def create_hubverse_table(model_fit_path: Path) -> None:
    """Create hubverse table from model fit using R script.

    Parameters
    ----------
    model_fit_path : Path
        Path to the model fit directory.

    Returns
    -------
    None
    """
    run_r_code(
        f"""
            forecasttools::write_tabular(
            hewr::model_fit_dir_to_hub_q_tbl('{model_fit_path}'),
            fs::path('{model_fit_path}', "hubverse_table", ext = "parquet")
            )
            """,
        function_name="create_hubverse_table",
        text=True,
    )
    return None
