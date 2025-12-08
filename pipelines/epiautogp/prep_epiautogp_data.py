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


def convert_to_epiautogp_json(
    target: str,
    data_for_model_fit_path: Path,
    epiweekly_data_path: Path,
    output_json_path: Path,
    disease: str,
    location: str,
    forecast_date: dt.date,
    nowcast_dates: list[dt.date] | None = None,
    nowcast_reports: list[list[float]] | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Convert surveillance data to EpiAutoGP JSON format.

    This function reads surveillance data from the pyrenew-hew pipeline
    and converts it to the JSON format required by EpiAutoGP. It handles
    NSSP (emergency department visits) as weekly percentages of all ED visits
    and NHSN (hospital admissions) as weekly counts.

    Parameters
    ----------
    target : str
        Target data type: "nssp" for ED visit percentages or
        "nhsn" for hospital admission counts
    data_for_model_fit_path : Path
        Path to the data_for_model_fit.json file produced by
        `process_and_save_loc_data` from `pipelines/prep_data.py`.
        This file is used for NHSN hospital admission data.
    epiweekly_data_path : Path
        Path to the epiweekly_combined_training_data.tsv file containing
        weekly aggregated NSSP data. This file is used for NSSP ED visit data.
    output_json_path : Path
        Path where the EpiAutoGP JSON file will be saved
    disease : str
        Disease name (e.g., "COVID-19", "Influenza", "RSV")
    location : str
        Location abbreviation (e.g., "CA", "US")
    forecast_date : dt.date
        The reference date from which forecasting begins
    nowcast_dates : list[dt.date] | None, default=None
        Dates requiring nowcasting (typically recent dates with
        incomplete data). If None, defaults to empty list. Not currently used.
    nowcast_reports : list[list[float]] | None, default=None
        Uncertainty bounds or samples for nowcast dates. If None,
        defaults to empty list. Not currently used.
    logger : logging.Logger | None, default=None
        Logger instance for logging messages. If None, a module-level
        logger will be created.

    Returns
    -------
    Path
        Path to the created EpiAutoGP JSON file

    Raises
    ------
    ValueError
        If target is not "nssp" or "nhsn", or if the required data
        is not present in the data_for_model_fit.json file
    FileNotFoundError
        If data_for_model_fit.json or epiweekly_combined_training_data.tsv
        doesn't exist

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

    # Set defaults for nowcasting
    if nowcast_dates is None:
        nowcast_dates = []
    if nowcast_reports is None:
        nowcast_reports = []

    # Read and process data based on target
    if target == "nssp":
        logger.info(f"Reading NSSP data from {epiweekly_data_path}")
        dates, reports = _extract_nssp_data(
            epiweekly_data_path, disease, location, logger
        )
    else:  # target == "nhsn"
        logger.info(f"Reading NHSN data from {data_for_model_fit_path}")
        dates, reports = _extract_nhsn_data(data_for_model_fit_path, location, logger)

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

    return output_json_path


def _extract_nssp_data(
    epiweekly_data_path: Path,
    disease: str,
    location: str,
    logger: logging.Logger,
) -> tuple[list[dt.date], list[float]]:
    """
    Extract NSSP ED visit percentage data from epiweekly TSV file.

    Parameters
    ----------
    epiweekly_data_path : Path
        Path to the epiweekly_combined_training_data.tsv file containing
        weekly NSSP data
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
    logger.info(f"Extracting weekly NSSP data for {disease} {location}")

    # Read the TSV file
    df = pl.read_csv(epiweekly_data_path, separator="\t")

    # Ensure date column is properly typed as Date
    df = df.with_columns(pl.col("date").cast(pl.Date))

    # Filter for the location, disease, and NSSP variables
    ed_data = df.filter(
        (pl.col("geo_value") == location)
        & (pl.col("disease") == disease)
        & (pl.col(".variable").is_in(["observed_ed_visits", "other_ed_visits"]))
    )

    if ed_data.height == 0:
        raise ValueError(
            f"No NSSP data found for {disease} {location} in {epiweekly_data_path}"
        )

    # Pivot the data to get observed_ed_visits and other_ed_visits as columns
    ed_data_pivoted = ed_data.pivot(
        index=["date", "geo_value", "disease"],
        on=".variable",
        values=".value",
    ).sort("date")

    # Calculate ED visit percentage
    ed_data_pivoted = ed_data_pivoted.with_columns(
        (
            pl.col("observed_ed_visits")
            / (pl.col("observed_ed_visits") + pl.col("other_ed_visits"))
            * 100.0
        ).alias("ed_visit_percentage")
    )

    dates = ed_data_pivoted["date"].to_list()
    reports = ed_data_pivoted["ed_visit_percentage"].to_list()

    logger.info(
        f"Extracted {len(dates)} weekly NSSP observations from {dates[0]} to {dates[-1]}"
    )

    return dates, reports


def _extract_nhsn_data(
    data_for_model_fit_path: Path, location: str, logger: logging.Logger
) -> tuple[list[dt.date], list[float]]:
    """
    Extract NHSN hospital admission counts from data_for_model_fit JSON file.

    Parameters
    ----------
    data_for_model_fit_path : Path
        Path to the data_for_model_fit.json file containing
        nhsn_training_data
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

    # Read the JSON file
    with open(data_for_model_fit_path, "r") as f:
        data_for_model_fit = json.load(f)

    # Get NHSN training data from the JSON
    nhsn_data = data_for_model_fit.get("nhsn_training_data")
    if nhsn_data is None:
        raise ValueError("No NHSN training data found in data_for_model_fit.json")

    # Convert to DataFrame
    df = pl.DataFrame(nhsn_data)

    # Ensure date column is properly typed as Date
    df = df.with_columns(pl.col("weekendingdate").cast(pl.Date))

    # Check if location matches (should be in jurisdiction column)
    if "jurisdiction" not in df.columns:
        # Data is already filtered for the location, no filtering needed
        hosp_data = df.sort("weekendingdate")
    else:
        hosp_data = df.filter(pl.col("jurisdiction") == location).sort("weekendingdate")

    if hosp_data.height == 0:
        raise ValueError(
            f"No NHSN data found for {location} in data_for_model_fit.json"
        )

    dates = hosp_data["weekendingdate"].to_list()
    reports = hosp_data["hospital_admissions"].cast(pl.Float64).to_list()

    logger.info(
        f"Extracted {len(dates)} NHSN observations from {dates[0]} to {dates[-1]}"
    )

    return dates, reports
