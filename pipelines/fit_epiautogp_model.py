"""
Functions for fitting EpiAutoGP models.

This module provides the interface for running EpiAutoGP Julia models,
following the same pattern as fit_pyrenew_model.py.
"""

from pathlib import Path

from pipelines.JuliaModel import JuliaModel


def fit_and_save_model(
    model_run_dir: str | Path,
    model_name: str,
    epiautogp_input_json: str | Path,
    n_forecast_weeks: int = 4,
    n_particles: int = 12,
    n_mcmc: int = 100,
    n_hmc: int = 50,
    n_forecast_draws: int = 10000,
    nthreads: int = 1,
) -> None:
    """
    Fit EpiAutoGP model and save results.

    This function runs the EpiAutoGP Julia model with the specified parameters
    and saves the forecast outputs to the model directory.

    Parameters
    ----------
    model_run_dir : str | Path
        Directory containing the model run data and where outputs will be saved.
    model_name : str
        Name of the model (typically "epiautogp").
    epiautogp_input_json : str | Path
        Path to the EpiAutoGP-formatted input JSON file.
    n_forecast_weeks : int, optional
        Number of weeks to forecast (default: 4).
    n_particles : int, optional
        Number of particles for filtering (default: 500).
    n_mcmc : int, optional
        Number of MCMC iterations (default: 500).
    n_hmc : int, optional
        Number of HMC iterations (default: 250).
    n_forecast_draws : int, optional
        Number of forecast draws (default: 10000).
    nthreads : int, optional
        Number of threads for Julia execution (default: 1).

    Raises
    ------
    FileNotFoundError
        If the Julia entrypoint script is not found.
    RuntimeError
        If the Julia model execution fails.

    Notes
    -----
    The model outputs will be saved to a subdirectory named `model_name`
    within `model_run_dir`. The main output is a hubverse-compatible
    forecast CSV file.
    """
    model_run_dir = Path(model_run_dir)
    epiautogp_input_json = Path(epiautogp_input_json)

    # Create model output directory
    model_output_dir = model_run_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Julia project and entrypoint paths
    julia_project_path = Path("EpiAutoGP")
    julia_entrypoint = julia_project_path / "run.jl"

    if not julia_entrypoint.exists():
        raise FileNotFoundError(
            f"Julia entrypoint script not found: {julia_entrypoint}"
        )

    # Initialize JuliaModel
    julia_model = JuliaModel(
        data_json_path=epiautogp_input_json,
        model_run_dir=model_output_dir,
        model_name=model_name,
        julia_project_path=julia_project_path,
        julia_entrypoint=julia_entrypoint,
        nthreads=nthreads,
    )

    # Prepare model parameters
    model_params = {
        "n-forecast-weeks": n_forecast_weeks,
        "n-particles": n_particles,
        "n-mcmc": n_mcmc,
        "n-hmc": n_hmc,
        "n-forecast-draws": n_forecast_draws,
    }

    # Run the model (this will raise RuntimeError if it fails)
    julia_model.run(model_params)

    return None
