import argparse
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpyro
import polars as pl
from prep_data import process_and_save_state

numpyro.set_host_device_count(4)

from fit_model import fit_and_save_model  # noqa
from generate_predictive import generate_and_save_predictions  # noqa


def forecast_denominator(
    model_run_dir: Path, n_forecast_days: int, n_samples: int
) -> None:
    subprocess.run(
        [
            "Rscript",
            "nssp_demo/forecast_non_target_visits.R",
            "--model-run-dir",
            f"{model_run_dir}",
            "--n-forecast-days",
            f"{n_forecast_days}",
            "--n-samples",
            f"{n_samples}",
        ]
    )
    return None


def postprocess_forecast(model_run_dir: Path) -> None:
    subprocess.run(
        [
            "Rscript",
            "nssp_demo/postprocess_state_forecast.R",
            "--model-run-dir",
            f"{model_run_dir}",
        ]
    )
    return None


def main(
    disease: str,
    report_date: str,
    state: str,
    facility_level_nssp_data_dir: Path | str,
    state_level_nssp_data_dir: Path | str,
    param_data_dir: Path | str,
    output_data_dir: Path | str,
    n_training_days: int,
    n_forecast_days: int,
    n_chains: int,
    n_warmup: int,
    n_samples: int,
    last_training_date: str = None,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    available_facility_level_reports = [
        datetime.strptime(f.stem, "%Y-%m-%d").date()
        for f in Path(facility_level_nssp_data_dir).glob("*.parquet")
    ]
    available_state_level_reports = [
        datetime.strptime(f.stem, "%Y-%m-%d").date()
        for f in Path(state_level_nssp_data_dir).glob("*.parquet")
    ]

    if report_date == "latest":
        report_date = max(available_facility_level_reports)
    else:
        report_date = datetime.strptime(report_date, "%Y-%m-%d").date()

    if report_date in available_state_level_reports:
        state_report_date = report_date
    else:
        state_report_date = max(available_state_level_reports)

    logger.info(f"Report date: {report_date}")
    logger.info(f"Using state-level data as of: {state_report_date}")

    if last_training_date == "latest":
        # + 1 because max date in dataset is report_date - 1
        last_training_date = report_date - timedelta(days=1)
    else:
        last_training_date = datetime.strptime(
            last_training_date, "%Y-%m-%d"
        ).date()

    if last_training_date >= report_date:
        raise ValueError(
            "Last training date must be before the report date. "
            "Got a last training date of {last_training_date} "
            "with a report date of {report_date}."
        )

    logger.info(f"last training date: {last_training_date}")

    # +1 because max date in dataset is report_date - 1
    first_training_date = last_training_date - timedelta(
        days=n_training_days - 1
    )

    logger.info(f"First training date {first_training_date}")

    facility_level_nssp_data, state_level_nssp_data = None, None

    if report_date in available_facility_level_reports:
        logger.info(
            "Facility level data available for " "the given report date"
        )
        facility_datafile = f"{report_date}.parquet"
        facility_level_nssp_data = pl.scan_parquet(
            Path(facility_level_nssp_data_dir, facility_datafile)
        )
    if state_report_date in available_state_level_reports:
        logger.info("State-level data available for the " "given report date.")
        state_datafile = f"{state_report_date}.parquet"
        state_level_nssp_data = pl.scan_parquet(
            Path(state_level_nssp_data_dir, state_datafile)
        )
    if facility_level_nssp_data is None and state_level_nssp_data is None:
        raise ValueError(
            "No data available for the requested report date " f"{report_date}"
        )

    param_estimates = pl.scan_parquet(Path(param_data_dir, "prod.parquet"))
    model_batch_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )

    model_batch_dir = Path(output_data_dir, model_batch_dir_name)

    model_run_dir = Path(model_batch_dir, state)

    os.makedirs(model_run_dir, exist_ok=True)

    logger.info(f"Processing {state}")
    process_and_save_state(
        state_abb=state,
        disease=disease,
        facility_level_nssp_data=facility_level_nssp_data,
        state_level_nssp_data=state_level_nssp_data,
        report_date=report_date,
        state_level_report_date=state_report_date,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        param_estimates=param_estimates,
        model_batch_dir=model_batch_dir,
        logger=logger,
    )
    logger.info("Data preparation complete.")

    logger.info("Fitting model")
    fit_and_save_model(
        model_run_dir,
        n_warmup=n_warmup,
        n_samples=n_samples,
        n_chains=n_chains,
    )
    logger.info("Model fitting complete")

    logger.info("Performing posterior prediction / forecasting...")
    generate_and_save_predictions(model_run_dir, n_forecast_days)

    logger.info("Performing non-target pathogen forecasting...")
    n_denominator_samples = n_samples * n_chains
    forecast_denominator(model_run_dir, n_forecast_days, n_denominator_samples)
    logger.info("Forecasting complete.")

    logger.info("Postprocessing forecast...")
    postprocess_forecast(model_run_dir)
    logger.info("Postprocessing complete.")
    logger.info(
        "Single state pipeline complete "
        f"for state {state} with "
        f"report date {report_date}."
    )

    return None


parser = argparse.ArgumentParser(
    description="Create fit data for disease modeling."
)
parser.add_argument(
    "--disease",
    type=str,
    required=True,
    help="Disease to model (e.g., COVID-19, Influenza, RSV).",
)

parser.add_argument(
    "--state",
    type=str,
    required=True,
    help=(
        "Two letter abbreviation for the state to fit"
        "(e.g. 'AK', 'AL', 'AZ', etc.)."
    ),
)

parser.add_argument(
    "--report-date",
    type=str,
    default="latest",
    help="Report date in YYYY-MM-DD format or latest (default: latest).",
)

parser.add_argument(
    "--facility-level-nssp-data-dir",
    type=Path,
    default=Path("private_data", "nssp_etl_gold"),
    help=(
        "Directory in which to look for facility-level NSSP " "ED visit data"
    ),
)

parser.add_argument(
    "--state-level-nssp-data-dir",
    type=Path,
    default=Path("private_data", "nssp_state_level_gold"),
    help=("Directory in which to look for state-level NSSP " "ED visit data."),
)

parser.add_argument(
    "--param-data-dir",
    type=Path,
    default=Path("private_data", "prod_param_estimates"),
    help=(
        "Directory in which to look for parameter estimates"
        "such as delay PMFs."
    ),
)

parser.add_argument(
    "--output-data-dir",
    type=Path,
    default="private_data",
    help="Directory in which to save output data.",
)

parser.add_argument(
    "--n-training-days",
    type=int,
    default=180,
    help="Number of training days (default: 180).",
)

parser.add_argument(
    "--n-forecast-days",
    type=int,
    default=28,
    help="Number of days ahead to forecast (default: 28).",
)


parser.add_argument(
    "--n-chains",
    type=int,
    default=4,
    help="Number of MCMC chains to run (default: 4).",
)

parser.add_argument(
    "--n-warmup",
    type=int,
    default=1000,
    help=("Number of warmup iterations per chain for NUTS" "(default: 1000)."),
)

parser.add_argument(
    "--n-samples",
    type=int,
    default=1000,
    help=(
        "Number of posterior samples to draw per "
        "chain using NUTS (default: 1000)."
    ),
)

parser.add_argument(
    "--last-training-date",
    type=str,
    default="latest",
    help=(
        "Last date to use for model training in "
        "YYYY-MM-DD format or 'latest' (default: latest)."
    ),
)


if __name__ == "__main__":
    args = parser.parse_args()
    numpyro.set_host_device_count(args.n_chains)
    main(**vars(args))
