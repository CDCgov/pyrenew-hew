"""
Data preparation and conversion functions for EpiAutoGP.

This module provides functions to convert surveillance data from the
pyrenew-hew pipeline format to the JSON format expected by EpiAutoGP.
"""

import datetime as dt
import json
import logging
from pathlib import Path

import polars as pl

from pipelines.epiautogp.epiautogp_forecast_utils import (
    ForecastPipelineContext,
    ModelPaths,
)


def _validate_epiautogp_parameters(
    target: str,
    frequency: str,
    use_percentage: bool,
) -> None:
    """
    Validate EpiAutoGP conversion parameters.

    The inadmissible parameter combinations are:
    - `target` not in ['nssp', 'nhsn']
    - `frequency` not in ['daily', 'epiweekly']
    - `use_percentage` is `True` when target is 'nhsn' (NHSN data are always counts)
    - `frequency` is 'daily' when target is 'nhsn' (NHSN data are only epiweekly)
    """
    # Validate individual parameters
    if target not in ["nssp", "nhsn"]:
        raise ValueError(f"target must be 'nssp' or 'nhsn', got '{target}'")

    if frequency not in ["daily", "epiweekly"]:
        raise ValueError(f"frequency must be 'daily' or 'epiweekly', got '{frequency}'")

    # Validate parameter combinations
    if target == "nhsn" and use_percentage:
        raise ValueError(
            "use_percentage is only applicable when target='nssp'. "
            "NHSN hospital admissions are always reported as counts."
        )

    if target == "nhsn" and frequency == "daily":
        raise ValueError("NHSN data is only available in epiweekly frequency.")


def convert_to_epiautogp_json(
    context: ForecastPipelineContext,
    paths: ModelPaths,
    nowcast_dates: list[dt.date] | None = None,
    nowcast_reports: list[list[float]] | None = None,
) -> Path:
    """
    Convert surveillance data to EpiAutoGP JSON format.

    This function reads surveillance data from the pyrenew-hew pipeline
    and converts it to the JSON format required by EpiAutoGP. It supports
    both daily and epiweekly data, and can output either counts or percentages.

    Parameters
    ----------
    context : ForecastPipelineContext
        Forecast pipeline context containing disease, location, report_date, and logger
    paths : ModelPaths
        Model paths containing daily and epiweekly training data paths
    output_json_path : Path
        Path where the EpiAutoGP JSON file will be saved
    target : str, default="nssp"
        Target data type: "nssp" for ED visit data or
        "nhsn" for hospital admission counts
    frequency : str, default="epiweekly"
        Data frequency: "daily" or "epiweekly"
    use_percentage : bool, default=False
        If True, convert ED visits to percentage:
        observed_ed_visits / (observed_ed_visits + other_ed_visits) * 100
        Only applicable for NSSP target.
    nowcast_dates : Optional[list[dt.date]], default=None
        Dates requiring nowcasting (typically recent dates with
        incomplete data). If None, defaults to empty list. Not currently used.
    nowcast_reports : Optional[list[list[float]]], default=None
        Uncertainty bounds or samples for nowcast dates. If None,
        defaults to empty list. Not currently used.

    Returns
    -------
    Path
        Path to the created EpiAutoGP JSON file

    Raises
    ------
    ValueError
        If target is not "nssp" or "nhsn", if frequency is not "daily" or
        "epiweekly", if use_percentage is True when target is "nhsn",
        or if the required data is not present
    FileNotFoundError
        If data files don't exist

    Required Output Structure
    -----
    The output JSON for `EpiAutoGP` must have the following structure:
    {
        "dates": ["2024-01-01", "2024-01-02", ...],
        "reports": [45.0, 52.0, ...],
        "pathogen": "COVID-19",
        "location": "CA",
        "target": "nssp",
        "forecast_date": "2024-01-02",
        "nowcast_dates": [], # eventually vector of dates for nowcasting
        "nowcast_reports": [] # eventually vector of vectors for nowcast uncertainty
    }
    """
    logger = context.logger

    # Validate parameters
    _validate_epiautogp_parameters(
        context.target, context.frequency, context.use_percentage
    )

    # Set defaults for nowcasting
    if nowcast_dates is None:
        nowcast_dates = []
    if nowcast_reports is None:
        nowcast_reports = []

    # Define input data JSON path
    input_json_path = paths.model_output_dir / f"{context.model_name}_input.json"
    # Determine which data path to use based on frequency
    if context.frequency == "daily":
        data_path = paths.daily_training_data
    else:  # epiweekly
        data_path = paths.epiweekly_training_data

    # Read data from TSV
    logger.info(f"Reading {context.frequency} data from {data_path}")
    dates, reports = _read_tsv_data(
        data_path,
        context.disease,
        context.loc,
        context.target,
        context.frequency,
        context.use_percentage,
        logger,
    )

    # Create EpiAutoGP input structure
    epiautogp_input = {
        "dates": [d.isoformat() for d in dates],
        "reports": reports,
        "pathogen": context.disease,
        "location": context.loc,
        "target": context.target,
        "use_percentage": context.use_percentage,
        "forecast_date": context.report_date.isoformat(),
        "nowcast_dates": [d.isoformat() for d in nowcast_dates],
        "nowcast_reports": nowcast_reports,
    }

    # Write JSON file
    input_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_json_path, "w") as f:
        json.dump(epiautogp_input, f, indent=2)

    logger.info(
        f"Saved EpiAutoGP input JSON for {context.disease} {context.loc} "
        f"(target={context.target}) to {input_json_path}"
    )

    return input_json_path


def _read_tsv_data(
    tsv_path: Path,
    disease: str,
    location: str,
    target: str,
    frequency: str,
    use_percentage: bool,
    logger: logging.Logger,
) -> tuple[list[dt.date], list[float]]:
    """
    Read surveillance data from TSV files.
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    logger.info(f"Reading {frequency} data from {tsv_path}")

    # Read TSV file
    df = pl.read_csv(tsv_path, separator="\t")

    # Filter for the specified location and disease (case-insensitive for disease)
    df = df.filter(
        (pl.col("geo_value") == location)
        & (pl.col("disease").str.to_uppercase() == disease.upper())
    )

    if df.height == 0:
        raise ValueError(f"No data found for {disease} {location} in {tsv_path}")

    # Pivot the data to get columns for each variable
    df_pivot = df.pivot(
        index=["date", "geo_value", "disease", "data_type"],
        on=".variable",
        values=".value",
    )

    # Ensure date column is properly typed
    df_pivot = df_pivot.with_columns(pl.col("date").cast(pl.Date))
    df_pivot = df_pivot.sort("date")

    # Extract data based on target
    if target == "nssp":
        dates, reports = _extract_nssp_from_pivot(
            df_pivot, use_percentage, tsv_path, logger
        )
    else:  # target == "nhsn"
        dates, reports = _extract_nhsn_from_pivot(df_pivot, tsv_path, logger)

    logger.info(
        f"Extracted {len(dates)} {frequency} {target} observations "
        f"from {dates[0]} to {dates[-1]}"
    )

    return dates, reports


def _extract_nssp_from_pivot(
    df_pivot: pl.DataFrame,
    use_percentage: bool,
    tsv_path: Path,
    logger: logging.Logger,
) -> tuple[list[dt.date], list[float]]:
    """
    Extract NSSP ED visit data from pivoted DataFrame.
    """
    # Check required columns
    if "observed_ed_visits" not in df_pivot.columns:
        raise ValueError(f"Column 'observed_ed_visits' not found in {tsv_path}")

    if use_percentage:
        if "other_ed_visits" not in df_pivot.columns:
            raise ValueError(
                f"Column 'other_ed_visits' required for percentage calculation but not found in {tsv_path}"
            )
        # Calculate percentage
        df_pivot = df_pivot.with_columns(
            (
                pl.col("observed_ed_visits")
                / (pl.col("observed_ed_visits") + pl.col("other_ed_visits"))
                * 100.0
            ).alias("value")
        )
        logger.info(
            "Using ED visit percentage (observed_ed_visits / total_ed_visits * 100)"
        )
    else:
        # Use raw counts
        df_pivot = df_pivot.with_columns(
            pl.col("observed_ed_visits").cast(pl.Float64).alias("value")
        )
        logger.info("Using raw ED visit counts")

    # Filter out any rows with null values
    df_pivot = df_pivot.filter(pl.col("value").is_not_null())

    dates = df_pivot["date"].to_list()
    reports = df_pivot["value"].to_list()

    return dates, reports


def _extract_nhsn_from_pivot(
    df_pivot: pl.DataFrame,
    tsv_path: Path,
    logger: logging.Logger,
) -> tuple[list[dt.date], list[float]]:
    """
    Extract NHSN hospital admission data from pivoted DataFrame.
    """
    if "observed_hospital_admissions" not in df_pivot.columns:
        raise ValueError(
            f"Column 'observed_hospital_admissions' not found in {tsv_path}"
        )

    df_pivot = df_pivot.with_columns(
        pl.col("observed_hospital_admissions").cast(pl.Float64).alias("value")
    )
    logger.info("Using hospital admission counts")

    # Filter out any rows with null values
    df_pivot = df_pivot.filter(pl.col("value").is_not_null())

    dates = df_pivot["date"].to_list()
    reports = df_pivot["value"].to_list()

    return dates, reports
