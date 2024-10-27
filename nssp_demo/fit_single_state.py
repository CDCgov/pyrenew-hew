import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from prep_data import process_and_save_state

import polars as pl
import numpyro
numpyro.set_host_device_count(4)

from fit_model import fit_and_save_model # noqa
from generate_predictive import generate_and_save_predictions # noqa


def main(
        disease,
        report_date,
        state,
        nssp_data_dir,
        param_data_dir,
        output_data_dir,
        n_training_days,
        n_forecast_days,
        n_chains,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if report_date == "latest":
        report_date = max(
            f.stem for f in Path(nssp_data_dir).glob("*.parquet")
        )

    report_date = datetime.strptime(report_date, "%Y-%m-%d").date()

    logger.info(f"Report date: {report_date}")

    last_training_date = report_date - timedelta(days=1)
    # + 1 because max date in dataset is report_date - 1
    first_training_date = last_training_date - timedelta(
        days=n_training_days - 1
    )

    datafile = f"{report_date}.parquet"
    nssp_data = pl.scan_parquet(Path(nssp_data_dir, datafile))
    param_estimates = pl.scan_parquet(Path(param_data_dir, "prod.parquet"))
    model_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )

    model_data_dir = Path(output_data_dir, model_dir_name)

    model_fit_dir = Path(model_data_dir, state)

    os.makedirs(model_data_dir, exist_ok=True)

    logger.info(f"Processing {state}")
    process_and_save_state(
        state_abb=state,
        disease=disease,
        nssp_data=nssp_data,
        report_date=report_date,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        param_estimates=param_estimates,
        model_data_dir=model_data_dir,
        logger=logger,
    )
    logger.info("Data preparation complete.")

    logger.info("Fitting model")
    fit_and_save_model(model_fit_dir)
    logger.info("Model fitting complete")

    logger.info("Performing posterior prediction / forecasting...")
    generate_and_save_predictions(
        model_fit_dir,
        n_forecast_days)

    return None


parser = argparse.ArgumentParser(
    description="Create fit data for disease modeling."
)
parser.add_argument(
    "--disease",
    type=str,
    required=True,
    help="Disease to model (e.g., COVID-19, Influenza, RSV)",
)

parser.add_argument(
    "--state",
    type=str,
    required=True,
    help=("Two letter abbreviation for the state to fit"
          "(e.g. 'AK', 'AL', 'AZ', etc.)"))

parser.add_argument(
    "--report-date",
    type=str,
    default="latest",
    help="Report date in YYYY-MM-DD format or latest (default: latest)",
)

parser.add_argument(
    "--nssp-data-dir",
    type=Path,
    default=Path("private_data", "nssp_etl_gold"),
    help="Directory in which to look for NSSP input data.",
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
    help="Number of training days (default: 180)",
)

parser.add_argument(
    "--n-forecast-days",
    type=int,
    default=28,
    help="Number of days ahead to forecast")

parser.add_argument(
    "--n-chains",
    type=int,
    default=4,
    help="Number of MCMC chains to run (default: 4)")


if __name__ == "__main__":
    args = parser.parse_args()
    numpyro.set_host_device_count(args.n_chains)
    main(**vars(args))
