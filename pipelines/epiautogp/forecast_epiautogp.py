import argparse
import logging
from pathlib import Path

from pipelines.cli_utils import add_common_forecast_arguments
from pipelines.common_utils import (
    create_hubverse_table,
    run_julia_code,
    run_julia_script,
    run_r_script,
)
from pipelines.epiautogp import convert_to_epiautogp_json
from pipelines.epiautogp.process_epiautogp_forecast import process_epiautogp_forecast
from pipelines.forecast_utils import (
    prepare_model_data,
    setup_forecast_pipeline,
)


def run_epiautogp_forecast(
    json_input_path: Path,
    model_dir: Path,
    target: str,
    n_forecast_weeks: int = 8,
    n_particles: int = 24,
    n_mcmc: int = 100,
    n_hmc: int = 50,
    n_forecast_draws: int = 2000,
    transformation: str = "boxcox",
    smc_data_proportion: float = 0.1,
) -> None:
    """
    Run EpiAutoGP forecasting model using Julia.

    Parameters
    ----------
    model_dir : Path
        Directory containing the model data and where outputs will be saved
    target : str
        Target data type: "nssp" for ED visit data or "nhsn" for hospital admissions
    n_forecast_weeks : int, default=8
        Number of weeks to forecast
    n_particles : int, default=24
        Number of particles for SMC
    n_mcmc : int, default=100
        Number of MCMC steps for GP kernel structure
    n_hmc : int, default=50
        Number of HMC steps for GP kernel hyperparameters
    n_forecast_draws : int, default=2000
        Number of forecast draws
    transformation : str, default="boxcox"
        Data transformation type
    smc_data_proportion : float, default=0.1
        Proportion of data used in each SMC step
    """
    # Use model_dir directly (not a subdirectory) to match R pipeline expectations
    # The R plotting code expects parquet files at model_dir/filename.parquet
    output_dir = Path(model_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate julia environment for EpiAutoGP
    run_julia_code(
        """
        using Pkg
        Pkg.activate("EpiAutoGP")
        Pkg.instantiate()
        """,
        function_name="setup_epiautogp_environment",
    )

    # Run Julia script
    run_julia_script(
        "EpiAutoGP/run.jl",
        [
            f"--json-input={json_input_path}",
            f"--output-dir={output_dir}",
            f"--n-forecast-weeks={n_forecast_weeks}",
            f"--n-particles={n_particles}",
            f"--n-mcmc={n_mcmc}",
            f"--n-hmc={n_hmc}",
            f"--n-forecast-draws={n_forecast_draws}",
            f"--transformation={transformation}",
            f"--smc-data-proportion={smc_data_proportion}",
        ],
        executor_flags=["--project=EpiAutoGP"],
        function_name="run_epiautogp_forecast",
    )
    return None


def main(
    disease: str,
    report_date: str,
    loc: str,
    facility_level_nssp_data_dir: Path | str,
    state_level_nssp_data_dir: Path | str,
    param_data_dir: Path | str,
    output_dir: Path | str,
    n_training_days: int,
    n_forecast_days: int,
    target: str,
    frequency: str,
    use_percentage: bool = False,
    exclude_last_n_days: int = 0,
    eval_data_path: Path = None,
    credentials_path: Path = None,
    nhsn_data_path: Path = None,
    n_forecast_weeks: int = 8,
    n_particles: int = 24,
    n_mcmc: int = 100,
    n_hmc: int = 50,
    n_forecast_draws: int = 2000,
    transformation: str = "boxcox",
    smc_data_proportion: float = 0.1,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Generate model name
    model_name = f"epiautogp_{target}"
    if use_percentage:
        model_name += "_pct"

    logger.info(
        "Starting single-location EpiAutoGP forecasting pipeline for "
        f"location {loc}, and report date {report_date}"
    )

    # Step 1: Setup pipeline (loads data, validates dates, creates directories)
    # This is the context of the forecast pipeline
    context = setup_forecast_pipeline(
        disease=disease,
        report_date=report_date,
        loc=loc,
        facility_level_nssp_data_dir=facility_level_nssp_data_dir,
        state_level_nssp_data_dir=state_level_nssp_data_dir,
        output_dir=output_dir,
        n_training_days=n_training_days,
        n_forecast_days=n_forecast_days,
        exclude_last_n_days=exclude_last_n_days,
        credentials_path=credentials_path,
        logger=logger,
    )

    # Step 2: Prepare data (process location data, eval data, epiweekly data)
    # returns paths to prepared data files and directories
    paths = prepare_model_data(
        context=context,
        model_name=model_name,
        eval_data_path=eval_data_path,
        nhsn_data_path=nhsn_data_path,
    )

    # Step 3: Convert data to EpiAutoGP JSON format
    logger.info("Converting data to EpiAutoGP JSON format...")
    epiautogp_json_path = Path(paths.data_dir, f"epiautogp_input_{target}.json")

    epiautogp_json_path = convert_to_epiautogp_json(
        daily_training_data_path=paths.daily_training_data,
        epiweekly_training_data_path=paths.epiweekly_training_data,
        output_json_path=epiautogp_json_path,
        disease=disease,
        location=loc,
        forecast_date=context.report_date,
        target=target,
        frequency=frequency,
        use_percentage=use_percentage,
        logger=logger,
    )

    # Step 4: Run EpiAutoGP forecast
    logger.info("Performing EpiAutoGP forecasting...")
    run_epiautogp_forecast(
        json_input_path=epiautogp_json_path,
        model_dir=paths.model_output_dir,
        target=target,
        n_forecast_weeks=n_forecast_weeks,
        n_particles=n_particles,
        n_mcmc=n_mcmc,
        n_hmc=n_hmc,
        n_forecast_draws=n_forecast_draws,
        transformation=transformation,
        smc_data_proportion=smc_data_proportion,
    )

    # Step 5: Process forecast outputs (combine with observed, calculate CIs)
    logger.info("Processing forecast outputs...")
    process_epiautogp_forecast(
        model_run_dir=context.model_run_dir,
        model_name=model_name,
        target=target,
        n_forecast_days=context.n_forecast_days,
        save=True,
    )
    logger.info("Forecast processing complete.")

    # Step 6: Create hubverse table
    logger.info("Creating hubverse table...")
    create_hubverse_table(Path(context.model_run_dir, model_name))
    logger.info("Postprocessing complete.")

    # Step 7: Generate forecast plots
    logger.info("Generating forecast plots...")
    plot_script = Path(__file__).parent / "plot_epiautogp_forecast.R"
    run_r_script(
        str(plot_script),
        [context.model_run_dir, "--epiautogp-model-name", model_name],
        capture_output=False,
        text=True,
    )
    logger.info("Plotting complete.")

    logger.info(
        "Single-location EpiAutoGP pipeline complete "
        f"for location {loc}, and "
        f"report date {report_date}."
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EpiAutoGP forecasting pipeline for disease modeling."
    )

    # Add common arguments
    add_common_forecast_arguments(parser)

    # Add EpiAutoGP-specific arguments
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["nssp", "nhsn"],
        help=(
            "Target data type: 'nssp' for ED visit data or "
            "'nhsn' for hospital admission counts."
        ),
    )

    parser.add_argument(
        "--frequency",
        type=str,
        default="epiweekly",
        choices=["daily", "epiweekly"],
        help="Data frequency: 'daily' or 'epiweekly' (default: epiweekly).",
    )

    parser.add_argument(
        "--use-percentage",
        action="store_true",
        help=(
            "Convert ED visits to percentage of total ED visits "
            "(only applicable for NSSP target)."
        ),
    )

    parser.add_argument(
        "--n-forecast-weeks",
        type=int,
        default=8,
        help="Number of weeks to forecast (default: 8).",
    )

    parser.add_argument(
        "--n-particles",
        type=int,
        default=24,
        help="Number of particles for SMC (default: 24).",
    )

    parser.add_argument(
        "--n-mcmc",
        type=int,
        default=100,
        help="Number of MCMC steps for GP kernel structure (default: 100).",
    )

    parser.add_argument(
        "--n-hmc",
        type=int,
        default=50,
        help="Number of HMC steps for GP kernel hyperparameters (default: 50).",
    )

    parser.add_argument(
        "--n-forecast-draws",
        type=int,
        default=2000,
        help="Number of forecast draws (default: 2000).",
    )

    parser.add_argument(
        "--transformation",
        type=str,
        default="boxcox",
        help="Data transformation type (default: boxcox).",
    )

    parser.add_argument(
        "--smc-data-proportion",
        type=float,
        default=0.1,
        help="Proportion of data used in each SMC step (default: 0.1).",
    )

    args = parser.parse_args()
    main(**vars(args))
