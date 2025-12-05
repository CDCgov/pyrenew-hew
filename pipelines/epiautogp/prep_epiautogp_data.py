"""
Data preparation and conversion functions for EpiAutoGP.

This module provides functions to convert surveillance data from the
pyrenew-hew pipeline format to the JSON format expected by EpiAutoGP.
"""

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

import polars as pl


def convert_to_epiautogp_json(
    combined_training_data_path: Path,
    nhsn_data_path: Optional[Path],
    output_json_path: Path,
    disease: str,
    location: str,
    forecast_date: dt.date,
    target: str = "nssp",
    nowcast_dates: Optional[list[dt.date]] = None,
    nowcast_reports: Optional[list[list[float]]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Convert surveillance data to EpiAutoGP JSON format.

    This function reads surveillance data from the pyrenew-hew pipeline
    and converts it to the JSON format required by EpiAutoGP. It handles
    NSSP (emergency department visits) as weekly percentages of all ED visits
    and NHSN (hospital admissions) as weekly counts.

    Parameters
    ----------
    combined_training_data_path : Path
        Path to the combined_training_data.tsv file produced by
        `process_and_save_loc_data` from `pipelines/prep_data.py`
    nhsn_data_path : Optional[Path]
        Path to NHSN parquet file with hospital admission data.
        Required when target='nhsn'. Can be None when target='nssp'.
    output_json_path : Path
        Path where the EpiAutoGP JSON file will be saved
    disease : str
        Disease name (e.g., "COVID-19", "Influenza", "RSV")
    location : str
        Location abbreviation (e.g., "CA", "US")
    forecast_date : dt.date
        The reference date from which forecasting begins
    target : str, default="nssp"
        Target data type: "nssp" for ED visit percentages or
        "nhsn" for hospital admission counts
    nowcast_dates : Optional[list[dt.date]], default=None
        Dates requiring nowcasting (typically recent dates with
        incomplete data). If None, defaults to empty list. Not currently used.
    nowcast_reports : Optional[list[list[float]]], default=None
        Uncertainty bounds or samples for nowcast dates. If None,
        defaults to empty list. Not currently used.
    logger : Optional[logging.Logger], default=None
        Logger instance for logging messages. If None, a module-level
        logger will be created.

    Returns
    -------
    None
        Writes JSON file to output_json_path

    Raises
    ------
    ValueError
        If target is not "nssp" or "nhsn", or if nhsn_data_path is
        None when target="nhsn"
    FileNotFoundError
        If required input files don't exist

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
    if logger is None:
        logger = logging.getLogger(__name__)

    # Validate target
    if target not in ["nssp", "nhsn"]:
        raise ValueError(f"target must be 'nssp' or 'nhsn', got '{target}'")

    # Validate NHSN requirements
    if target == "nhsn" and nhsn_data_path is None:
        raise ValueError("nhsn_data_path is required when target='nhsn'")

    # Set defaults for nowcasting
    if nowcast_dates is None:
        nowcast_dates = []
    if nowcast_reports is None:
        nowcast_reports = []

    # Read and process data based on target
    if target == "nssp":
        dates, reports = _extract_nssp_data(
            combined_training_data_path, disease, location, logger
        )
    else:  # target == "nhsn"
        dates, reports = _extract_nhsn_data(nhsn_data_path, location, logger)

    # Create EpiAutoGP input structure
    epiautogp_input = {
        "dates": [d.isoformat() for d in dates],
        "reports": reports,
        "pathogen": disease,
        "location": location,
        "target": target,
        "forecast_date": forecast_date.isoformat(),
        "nowcast_dates": [d.isoformat() for d in nowcast_dates],
        "nowcast_reports": nowcast_reports,
    }

    # Write JSON file
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(epiautogp_input, f, indent=2)

    logger.info(
        f"Saved EpiAutoGP input JSON for {disease} {location} "
        f"(target={target}) to {output_json_path}"
    )


def _extract_nssp_data(
    combined_training_data_path: Path,
    disease: str,
    location: str,
    logger: logging.Logger,
) -> tuple[list[dt.date], list[float]]:
    """
    Extract NSSP ED visit percentage data from combined training data.

    Parameters
    ----------
    combined_training_data_path : Path
        Path to combined_training_data.tsv
    disease : str
        Disease name
    location : str
        Location abbreviation
    logger : logging.Logger
        Logger instance

    Returns
    -------
    tuple[list[dt.date], list[float]]
        Lists of dates and corresponding ED visit percentages
    """
    logger.info(f"Extracting NSSP data for {disease} {location}")

    # Read combined training data
    df = pl.read_csv(combined_training_data_path, separator="\t")

    # Ensure date column is properly typed as Date
    df = df.with_columns(pl.col("date").cast(pl.Date))

    # Filter for relevant disease, location, and ED visit data
    # We need both observed_ed_visits and other_ed_visits to calculate percentage
    ed_data = (
        df.filter(
            (pl.col("disease") == disease)
            & (pl.col("geo_value") == location)
            & (pl.col(".variable").is_in(["observed_ed_visits", "other_ed_visits"]))
        )
        .pivot(
            index=["date", "geo_value", "disease"],
            on=".variable",
            values=".value",
        )
        .sort("date")
    )

    if ed_data.height == 0:
        raise ValueError(
            f"No NSSP data found for {disease} {location} in "
            f"{combined_training_data_path}"
        )

    # Calculate ED visit percentage
    ed_data = ed_data.with_columns(
        (
            pl.col("observed_ed_visits")
            / (pl.col("observed_ed_visits") + pl.col("other_ed_visits"))
            * 100.0
        ).alias("ed_visit_percentage")
    )

    dates = ed_data["date"].to_list()
    reports = ed_data["ed_visit_percentage"].to_list()

    logger.info(
        f"Extracted {len(dates)} NSSP observations from {dates[0]} to {dates[-1]}"
    )

    return dates, reports


def _extract_nhsn_data(
    nhsn_data_path: Path, location: str, logger: logging.Logger
) -> tuple[list[dt.date], list[float]]:
    """
    Extract NHSN hospital admission counts from parquet file.

    Parameters
    ----------
    nhsn_data_path : Path
        Path to NHSN parquet file
    location : str
        Location abbreviation
    logger : logging.Logger
        Logger instance

    Returns
    -------
    tuple[list[dt.date], list[float]]
        Lists of week-ending dates and corresponding admission counts
    """
    logger.info(f"Extracting NHSN data for {location}")

    # Read NHSN parquet file
    df = pl.read_parquet(nhsn_data_path)

    # Filter for location and sort by date
    hosp_data = df.filter(pl.col("jurisdiction") == location).sort("weekendingdate")

    if hosp_data.height == 0:
        raise ValueError(f"No NHSN data found for {location} in {nhsn_data_path}")

    dates = hosp_data["weekendingdate"].to_list()
    reports = hosp_data["hospital_admissions"].cast(pl.Float64).to_list()

    logger.info(
        f"Extracted {len(dates)} NHSN observations from {dates[0]} to {dates[-1]}"
    )

    return dates, reports
