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
    plot_and_save_loc_forecast,
)
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
    ed_visit_type: str
    model_name: str
    param_data_dir: Path | None
    eval_data_path: Path | None
    nhsn_data_path: Path | None
    report_date: date
    first_training_date: date
    last_training_date: date
    n_forecast_days: int
    exclude_last_n_days: int
    exclude_date_ranges: list[tuple[date, date]] | None
    model_batch_dir: Path
    model_run_dir: Path
    credentials_dict: dict[str, Any]
    facility_level_nssp_data: pl.LazyFrame
    loc_level_nssp_data: pl.LazyFrame
    logger: logging.Logger

    def prepare_model_data(self) -> ModelPaths:
        """
        Prepare training and evaluation data for a model.

        This function performs the data preparation steps that are common across
        all forecast pipelines:
        1. Create model output directory
        2. Process and save location data
        3. Save evaluation data
        4. Generate epiweekly datasets

        Returns
        -------
        ModelPaths
            Object containing all model output directory and file paths

        Raises
        ------
        ValueError
            If eval_data_path is None
        """
        # Create model output directory
        model_output_dir = Path(self.model_run_dir, self.model_name)
        data_dir = Path(model_output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        self.logger.info(f"Processing data for {self.loc}")

        # Process and save location data
        process_and_save_loc_data(
            loc_abb=self.loc,
            disease=self.disease,
            facility_level_nssp_data=self.facility_level_nssp_data,
            loc_level_nssp_data=self.loc_level_nssp_data,
            report_date=self.report_date,
            first_training_date=self.first_training_date,
            last_training_date=self.last_training_date,
            save_dir=data_dir,
            logger=self.logger,
            credentials_dict=self.credentials_dict,
            nhsn_data_path=self.nhsn_data_path,
        )

        # Save evaluation data
        self.logger.info("Getting eval data...")
        if self.eval_data_path is None:
            raise ValueError("No path to an evaluation dataset provided.")

        save_eval_data(
            loc=self.loc,
            disease=self.disease,
            first_training_date=self.first_training_date,
            last_training_date=self.last_training_date,
            latest_comprehensive_path=self.eval_data_path,
            output_data_dir=data_dir,
            last_eval_date=self.report_date + timedelta(days=self.n_forecast_days),
            credentials_dict=self.credentials_dict,
            nhsn_data_path=self.nhsn_data_path,
        )
        self.logger.info("Done getting eval data.")

        # Generate epiweekly datasets
        self.logger.info("Generating epiweekly datasets from daily datasets...")
        generate_epiweekly_data(data_dir)

        self.logger.info("Data preparation complete.")

        # Return structured paths object
        return ModelPaths(
            model_output_dir=model_output_dir,
            data_dir=data_dir,
            daily_training_data=Path(data_dir, "combined_training_data.tsv"),
            epiweekly_training_data=Path(
                data_dir, "epiweekly_combined_training_data.tsv"
            ),
        )

    def post_process_forecast(self) -> None:
        """
        Post-process forecast outputs: create hubverse table and generate plots.

        This function performs the final post-processing steps:
        1. Generate forecast plots using hewr via plot_and_save_loc_forecast
           (which also processes samples via hewr::process_loc_forecast)
        2. Create hubverse table from processed outputs

        The plot_and_save_loc_forecast function with model_name auto-detects
        the model type and dispatches to process_model_samples.epiautogp(),
        which reads Julia output samples, adds metadata, calculates credible
        intervals, and saves formatted outputs.

        Returns
        -------
        None
        """
        # Generate forecast plots and process samples using hewr
        # The model_name parameter triggers auto-detection and S3 dispatch to
        # process_model_samples.epiautogp() which handles Julia output format
        self.logger.info("Processing forecast and generating plots...")
        plot_and_save_loc_forecast(
            model_run_dir=self.model_run_dir,
            n_forecast_days=self.n_forecast_days,
            model_name=self.model_name,
        )
        self.logger.info("Processing and plotting complete.")

        # Create hubverse table from processed outputs
        self.logger.info("Creating hubverse table...")
        create_hubverse_table(Path(self.model_run_dir, self.model_name))
        self.logger.info("Postprocessing complete.")


def setup_forecast_pipeline(
    disease: str,
    report_date: str,
    loc: str,
    target: str,
    frequency: str,
    use_percentage: bool,
    ed_visit_type: str,
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
    exclude_date_ranges: list[tuple[date, date]] | None = None,
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
    target : str
        Target data type: "nssp" or "nhsn"
    frequency : str
        Data frequency: "daily" or "epiweekly"
    use_percentage : bool
        If True, use percentage values for ED visits (NSSP only)
    ed_visit_type : str
        Type of ED visits: "observed" or "other" (NSSP only)
    model_name : str
        Name of the model configuration
    param_data_dir : Path | None
        Directory containing parameter data
    eval_data_path : Path | None
        Path to evaluation data file
    nhsn_data_path : Path | None
        Path to NHSN hospital admission data
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
    exclude_date_ranges : list[tuple[date, date]] | None, default=None
        List of date ranges to exclude from training data (inclusive).
        Each tuple contains (start_date, end_date).
    credentials_path : Path | None, default=None
        Path to credentials file
    logger : logging.Logger | None, default=None
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
        ed_visit_type=ed_visit_type,
        model_name=model_name,
        param_data_dir=param_data_dir,
        eval_data_path=eval_data_path,
        nhsn_data_path=nhsn_data_path,
        report_date=report_date_parsed,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        n_forecast_days=n_forecast_days,
        exclude_last_n_days=exclude_last_n_days,
        exclude_date_ranges=exclude_date_ranges,
        model_batch_dir=model_batch_dir,
        model_run_dir=model_run_dir,
        credentials_dict=credentials_dict,
        facility_level_nssp_data=facility_level_nssp_data,
        loc_level_nssp_data=loc_level_nssp_data,
        logger=logger,
    )
