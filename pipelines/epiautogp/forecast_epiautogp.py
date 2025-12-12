import argparse
import logging
from pathlib import Path

from pipelines.cli_utils import add_common_forecast_arguments
from pipelines.common_utils import (
    run_julia_code,
    run_julia_script,
)
from pipelines.epiautogp import (
    convert_to_epiautogp_json,
    post_process_forecast,
    prepare_model_data,
    setup_forecast_pipeline,
)


def run_epiautogp_forecast(
    json_input_path: Path,
    model_dir: Path,
    params: dict,
    execution_settings: dict,
) -> None:
    """
    Run EpiAutoGP forecasting model using Julia.

    Parameters
    ----------
    json_input_path : Path
        Path to the JSON input file for EpiAutoGP.
    model_dir : Path
        Directory to save model outputs.
    params : dict
        Parameters to pass to EpiAutoGP.
    execution_settings : dict
        Execution settings for the Julia environment.
    """
    # Ensure output directory exists
    model_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate julia environment for EpiAutoGP
    run_julia_code(
        """
        using Pkg
        Pkg.activate("EpiAutoGP")
        Pkg.instantiate()
        """,
        function_name="setup_epiautogp_environment",
    )

    # Add path arguments to pass to EpiAutoGP
    params["json-input"] = str(json_input_path)
    params["output-dir"] = str(model_dir)

    # Convert Python dict keys (with underscores) to Julia CLI args (with hyphens)
    args_to_epiautogp = [
        f"--{key.replace('_', '-')}={value}" for key, value in params.items()
    ]
    executor_flags = [f"--{key}={value}" for key, value in execution_settings.items()]

    # Run Julia script
    run_julia_script(
        "EpiAutoGP/run.jl",
        args_to_epiautogp,
        executor_flags=executor_flags,
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
    smc_data_proportion: float = 0.1,
    n_threads: int = 1,
) -> None:
    # Step 0: Set up logging, model name and params to pass to epiautogp
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Generate model name
    model_name = f"epiautogp_{target}_{frequency}"
    if use_percentage:
        model_name += "_pct"

    # Declare transformation type
    if use_percentage:
        transformation = "percentage"
    else:
        transformation = "boxcox"

    # Epiautogp params and execution settings
    params = {
        "n_particles": n_particles,
        "n_mcmc": n_mcmc,
        "n_hmc": n_hmc,
        "n_forecast_draws": n_forecast_draws,
        "transformation": transformation,
        "smc_data_proportion": smc_data_proportion,
    }
    execution_settings = {
        "project": "EpiAutoGP",
        "threads": n_threads,
    }

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
        target=target,
        frequency=frequency,
        use_percentage=use_percentage,
        model_name=model_name,
        param_data_dir=param_data_dir,
        eval_data_path=eval_data_path,
        nhsn_data_path=nhsn_data_path,
        facility_level_nssp_data_dir=facility_level_nssp_data_dir,
        state_level_nssp_data_dir=state_level_nssp_data_dir,
        output_dir=output_dir,
        n_training_days=n_training_days,
        n_forecast_days=n_forecast_days,
        exclude_last_n_days=exclude_last_n_days,
        credentials_path=credentials_path,
        logger=logger,
    )

    # Step 2: Prepare data for modelling (process location data, eval data, epiweekly data)
    # returns paths to prepared data files and directories
    paths = prepare_model_data(
        context=context,
    )

    # Step 3: Convert data to EpiAutoGP JSON format
    logger.info("Converting data to EpiAutoGP JSON format...")
    epiautogp_input_json_path = convert_to_epiautogp_json(
        context=context,
        paths=paths,
    )

    # Step 4: Run EpiAutoGP forecast
    logger.info("Performing EpiAutoGP forecasting...")
    run_epiautogp_forecast(
        json_input_path=epiautogp_input_json_path,
        model_dir=paths.model_output_dir,
        params=params,
        execution_settings=execution_settings,
    )

    # Step 5: Post-process forecast outputs
    post_process_forecast(context=context)

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
        "--smc-data-proportion",
        type=float,
        default=0.1,
        help="Proportion of data used in each SMC step (default: 0.1).",
    )

    args = parser.parse_args()
    main(**vars(args))
