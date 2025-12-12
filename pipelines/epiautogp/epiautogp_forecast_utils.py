"""
Shared utilities for forecast pipeline scripts.

This module contains common functionality used across different forecast
pipelines (pyrenew, timeseries, epiautogp, etc.).
"""

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from pipelines.common_utils import (
    calculate_training_dates,
    create_hubverse_table,
    get_available_reports,
    load_credentials,
    load_nssp_data,
    parse_and_validate_report_date,
    run_r_script,
)
from pipelines.epiautogp.process_epiautogp_forecast import process_epiautogp_forecast
from pipelines.forecast_pyrenew import generate_epiweekly_data
from pipelines.prep_data import process_and_save_loc_data
from pipelines.prep_eval_data import save_eval_data


@dataclass
class ModelPaths:
    """
    Container for model output directory structure and file paths.

    This class holds all the computed output paths for a specific model run,
    making it easier to track where data and results are stored.
    """

    model_output_dir: Path
    data_dir: Path
    daily_training_data: Path
    epiweekly_training_data: Path


@dataclass
class ForecastPipelineContext:
    """
    Container for common forecast pipeline data, input configurations and
    the logger.

    This class holds all the shared state that gets passed around during
    a forecast pipeline run, reducing the number of parameters that need
    to be passed between functions.
    """

    disease: str
    loc: str
    target: str
    frequency: str
    use_percentage: bool
    model_name: str
    param_data_dir: Path | None
    eval_data_path: Path | None
    nhsn_data_path: Path | None
    report_date: date
    first_training_date: date
    last_training_date: date
    n_forecast_days: int
    exclude_last_n_days: int
    model_batch_dir: Path
    model_run_dir: Path
    credentials_dict: dict[str, Any]
    facility_level_nssp_data: pl.LazyFrame
    loc_level_nssp_data: pl.LazyFrame
    logger: logging.Logger


def setup_forecast_pipeline(
    disease: str,
    report_date: str,
    loc: str,
    target: str,
    frequency: str,
    use_percentage: bool,
    model_name: str,
    param_data_dir: Path | None,
    eval_data_path: Path | None,
    nhsn_data_path: Path | None,
    facility_level_nssp_data_dir: Path | str,
    state_level_nssp_data_dir: Path | str,
    output_dir: Path | str,
    n_training_days: int,
    n_forecast_days: int,
    exclude_last_n_days: int = 0,
    credentials_path: Path = None,
    logger: logging.Logger = None,
) -> ForecastPipelineContext:
    """
    Set up common forecast pipeline infrastructure.

    This function performs the initial setup steps that are common across
    all forecast pipelines:
    1. Load credentials
    2. Get available report dates
    3. Parse and validate the report date
    4. Calculate training dates
    5. Load NSSP data
    6. Create batch directory structure

    Parameters
    ----------
    disease : str
        Disease to model (e.g., "COVID-19", "Influenza", "RSV")
    report_date : str
        Report date in YYYY-MM-DD format or "latest"
    loc : str
        Two-letter USPS location abbreviation (e.g., "CA", "NY")
    facility_level_nssp_data_dir : Path | str
        Directory containing facility-level NSSP ED visit data
    state_level_nssp_data_dir : Path | str
        Directory containing state-level NSSP ED visit data
    output_dir : Path | str
        Root directory for output
    n_training_days : int
        Number of days of training data
    n_forecast_days : int
        Number of days ahead to forecast
    exclude_last_n_days : int, default=0
        Number of recent days to exclude from training
    credentials_path : Path, optional
        Path to credentials file
    logger : logging.Logger, optional
        Logger instance. If None, creates a new logger

    Returns
    -------
    ForecastPipelineContext
        Context object containing all setup information
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(
        f"Setting up forecast pipeline for {disease}, "
        f"location {loc}, report date {report_date}"
    )

    # Load credentials
    credentials_dict = load_credentials(credentials_path, logger)

    # Get available reports
    available_facility_level_reports = get_available_reports(
        facility_level_nssp_data_dir
    )
    available_loc_level_reports = get_available_reports(state_level_nssp_data_dir)

    # Parse and validate report date
    report_date_parsed, loc_report_date = parse_and_validate_report_date(
        report_date,
        available_facility_level_reports,
        available_loc_level_reports,
        logger,
    )

    # Calculate training dates
    first_training_date, last_training_date = calculate_training_dates(
        report_date_parsed,
        n_training_days,
        exclude_last_n_days,
        logger,
    )

    # Load NSSP data
    facility_level_nssp_data, loc_level_nssp_data = load_nssp_data(
        report_date_parsed,
        loc_report_date,
        available_facility_level_reports,
        available_loc_level_reports,
        facility_level_nssp_data_dir,
        state_level_nssp_data_dir,
        logger,
    )

    # Create model batch directory structure
    model_batch_dir_name = (
        f"{disease.lower()}_r_{report_date_parsed}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )
    model_batch_dir = Path(output_dir, model_batch_dir_name)
    model_run_dir = Path(model_batch_dir, "model_runs", loc)

    logger.info(f"Model batch directory: {model_batch_dir}")
    logger.info(f"Model run directory: {model_run_dir}")

    return ForecastPipelineContext(
        disease=disease,
        loc=loc,
        target=target,
        frequency=frequency,
        use_percentage=use_percentage,
        model_name=model_name,
        param_data_dir=param_data_dir,
        eval_data_path=eval_data_path,
        nhsn_data_path=nhsn_data_path,
        report_date=report_date_parsed,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        n_forecast_days=n_forecast_days,
        exclude_last_n_days=exclude_last_n_days,
        model_batch_dir=model_batch_dir,
        model_run_dir=model_run_dir,
        credentials_dict=credentials_dict,
        facility_level_nssp_data=facility_level_nssp_data,
        loc_level_nssp_data=loc_level_nssp_data,
        logger=logger,
    )


def prepare_model_data(
    context: ForecastPipelineContext,
) -> ModelPaths:
    """
    Prepare training and evaluation data for a model.

    This function performs the data preparation steps that are common across
    all forecast pipelines:
    1. Create model output directory
    2. Process and save location data
    3. Save evaluation data
    4. Generate epiweekly datasets

    Parameters
    ----------
    context : ForecastPipelineContext
        Pipeline context with shared configuration
    model_name : str
        Name of the model (used for directory naming)
    eval_data_path : Path, optional
        Path to evaluation dataset
    nhsn_data_path : Path, optional
        Path to NHSN data (for local testing)
    loc_level_nwss_data : pl.DataFrame, optional
        Wastewater surveillance data (for pyrenew models)

    Returns
    -------
    ModelPaths
        Object containing all model output directory and file paths

    Raises
    ------
    ValueError
        If eval_data_path is None
    """
    logger = context.logger

    # Create model output directory
    model_output_dir = Path(context.model_run_dir, context.model_name)
    data_dir = Path(model_output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Processing data for {context.loc}")

    # Process and save location data
    process_and_save_loc_data(
        loc_abb=context.loc,
        disease=context.disease,
        facility_level_nssp_data=context.facility_level_nssp_data,
        loc_level_nssp_data=context.loc_level_nssp_data,
        report_date=context.report_date,
        first_training_date=context.first_training_date,
        last_training_date=context.last_training_date,
        save_dir=data_dir,
        logger=logger,
        credentials_dict=context.credentials_dict,
        nhsn_data_path=context.nhsn_data_path,
    )

    # Save evaluation data
    logger.info("Getting eval data...")
    if context.eval_data_path is None:
        raise ValueError("No path to an evaluation dataset provided.")

    save_eval_data(
        loc=context.loc,
        disease=context.disease,
        first_training_date=context.first_training_date,
        last_training_date=context.last_training_date,
        latest_comprehensive_path=context.eval_data_path,
        output_data_dir=data_dir,
        last_eval_date=context.report_date + timedelta(days=context.n_forecast_days),
        credentials_dict=context.credentials_dict,
        nhsn_data_path=context.nhsn_data_path,
    )
    logger.info("Done getting eval data.")

    # Generate epiweekly datasets
    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly_data(data_dir)

    logger.info("Data preparation complete.")

    # Return structured paths object
    return ModelPaths(
        model_output_dir=model_output_dir,
        data_dir=data_dir,
        daily_training_data=Path(data_dir, "combined_training_data.tsv"),
        epiweekly_training_data=Path(data_dir, "epiweekly_combined_training_data.tsv"),
    )


def post_process_forecast(
    context: ForecastPipelineContext,
) -> None:
    """
    Post-process forecast outputs: process results, create hubverse table, and generate plots.

    This function performs the final post-processing steps:
    1. Process forecast outputs (add metadata, calculate CIs)
    2. Create hubverse table
    3. Generate forecast plots using EpiAutoGP-specific plotting script

    Parameters
    ----------
    context : ForecastPipelineContext
        Pipeline context with shared configuration

    Returns
    -------
    None
    """
    logger = context.logger

    # Process forecast outputs (add metadata, calculate CIs)
    logger.info("Processing forecast outputs...")
    process_epiautogp_forecast(
        model_run_dir=context.model_run_dir,
        model_name=context.model_name,
        target=context.target,
        save=True,
    )
    logger.info("Forecast processing complete.")

    # Create hubverse table
    logger.info("Creating hubverse table...")
    create_hubverse_table(Path(context.model_run_dir, context.model_name))
    logger.info("Postprocessing complete.")

    # Generate forecast plots using EpiAutoGP-specific plotting script
    logger.info("Generating forecast plots...")
    plot_script = Path(__file__).parent / "plot_epiautogp_forecast.R"
    run_r_script(
        str(plot_script),
        [str(context.model_run_dir), "--epiautogp-model-name", context.model_name],
        function_name="plot_epiautogp_forecast",
    )
    logger.info("Plotting complete.")
