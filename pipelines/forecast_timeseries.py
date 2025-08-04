import argparse
import datetime as dt
import logging
import subprocess
from pathlib import Path

from prep_data import get_training_dates_and_model_dir


def plot_and_save_loc_forecast(
    model_run_dir: Path,
    n_forecast_days: int,
    timeseries_model_name: str,
) -> None:
    command = [
        "Rscript",
        "pipelines/plot_and_save_loc_forecast.R",
        f"{model_run_dir}",
        "--n-forecast-days",
        f"{n_forecast_days}",
        "--timeseries-model-name",
        f"{timeseries_model_name}",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"plot_and_save_loc_forecast: {result.stderr.decode('utf-8')}"
        )
    return None


def timeseries_ensemble_forecasts(
    model_run_dir: Path, model_name: str, n_forecast_days: int, n_samples: int
) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/forecast_timeseries_ensemble.R",
            f"{model_run_dir}",
            "--model-name",
            f"{model_name}",
            "--n-forecast-days",
            f"{n_forecast_days}",
            "--n-samples",
            f"{n_samples}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"timeseries_ensemble_forecasts: {result.stderr.decode('utf-8')}"
        )
    return None


def cdc_flat_baseline_forecasts(
    model_run_dir: Path, model_name: str, n_forecast_days: int
) -> None:
    result = subprocess.run(
        [
            "Rscript",
            "pipelines/forecast_cdc_flat_baseline.R",
            f"{model_run_dir}",
            "--model-name",
            f"{model_name}",
            "--n-forecast-days",
            f"{n_forecast_days}",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cdc_flat_baseline_forecasts: {result.stderr.decode('utf-8')}"
        )
    return None


def create_hubverse_table(model_fit_path):
    result = subprocess.run(
        [
            "Rscript",
            "-e",
            f"""
            forecasttools::write_tabular(
            hewr::model_fit_dir_to_hub_q_tbl('{model_fit_path}'),
            fs::path('{model_fit_path}', "hubverse_table", ext = "parquet")
            )
            """,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"create_hubverse_table: {result.stderr}")
    return None


def main(
    disease: str,
    loc: str,
    report_date: str,
    output_dir: Path | str,
    n_training_days: int,
    n_forecast_days: int,
    n_denominator_samples: int,
    model_letters: str,
    exclude_last_n_days: int = 0,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if model_letters != "e":
        raise ValueError(
            "Only model_letters 'e' is supported for 'timeseries' model_family."
        )

    ensemble_model_name = f"ts_ensemble_{model_letters}"
    baseline_model_name = f"baseline_cdc_{model_letters}"

    logger.info(
        "Starting single-location timeseries forecasting pipeline for "
        f"location {loc}, and report date {report_date}"
    )

    report_date = dt.datetime.strptime(report_date, "%Y-%m-%d").date()
    logger.info(f"Report date: {report_date}")
    (_, _, model_run_dir) = get_training_dates_and_model_dir(
        report_date,
        exclude_last_n_days,
        n_training_days,
        disease,
        loc,
        output_dir,
    )

    logger.info("Performing baseline forecasting and postprocessing...")

    n_days_past_last_training = n_forecast_days + exclude_last_n_days
    cdc_flat_baseline_forecasts(
        model_run_dir, baseline_model_name, n_days_past_last_training
    )
    create_hubverse_table(Path(model_run_dir, baseline_model_name))

    logger.info("Performing timeseries ensemble forecasting")
    timeseries_ensemble_forecasts(
        model_run_dir,
        ensemble_model_name,
        n_days_past_last_training,
        n_denominator_samples,
    )
    plot_and_save_loc_forecast(
        model_run_dir, n_days_past_last_training, ensemble_model_name
    )
    create_hubverse_table(Path(model_run_dir, ensemble_model_name))

    logger.info("Postprocessing complete.")

    logger.info(
        "Single-location timeseries pipeline complete "
        f"for location {loc}, and "
        f"report date {report_date}."
    )
    return None


if __name__ == "__main__":
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
        "--loc",
        type=str,
        required=True,
        help=(
            "Two-letter USPS abbreviation for the location to fit"
            "(e.g. 'AK', 'AL', 'AZ', etc.)."
        ),
    )

    parser.add_argument(
        "--model-letters",
        type=str,
        default="e",
        help=(
            "Fit the model corresponding to the provided model letters (e.g. 'he', 'e', 'hew')."
        ),
        required=True,
    )

    parser.add_argument(
        "--report-date",
        type=str,
        default=dt.datetime.today().strftime("%Y-%m-%d"),
        help="Report date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="private_data",
        help="Directory in which to save output.",
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
        help=(
            "Number of days ahead to forecast relative to the "
            "report date (default: 28).",
        ),
    )

    parser.add_argument(
        "--n-chains",
        type=int,
        default=4,
        help="Number of MCMC chains to run (default: 4).",
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
        "--exclude-last-n-days",
        type=int,
        default=0,
        help=(
            "Optionally exclude the final n days of available training "
            "data (Default: 0, i.e. exclude no available data"
        ),
    )

    args = parser.parse_args()
    n_denominator_samples = args.n_samples * args.n_chains
    delattr(args, "n_samples")
    delattr(args, "n_chains")
    main(**vars(args), n_denominator_samples=n_denominator_samples)
