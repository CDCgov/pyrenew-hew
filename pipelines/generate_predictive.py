import argparse
import logging
import pickle
from pathlib import Path

import arviz as az
from build_pyrenew_model import (
    build_model_from_dir,
)


def generate_and_save_predictions(
    model_run_dir: str | Path, model_name: str, n_forecast_points: int
) -> None:
    logger = logging.getLogger(__name__)
    model_run_dir = Path(model_run_dir)
    model_dir = Path(model_run_dir, model_name)

    if not model_dir.exists():
        raise FileNotFoundError(f"The directory {model_dir} does not exist.")
    (my_model, my_data) = build_model_from_dir(model_run_dir)

    my_model._init_model(1, 1)
    fresh_sampler = my_model.mcmc.sampler

    with open(
        model_dir / "posterior_samples.pickle",
        "rb",
    ) as file:
        my_model.mcmc = pickle.load(file)

    my_model.mcmc.sampler = fresh_sampler
    logger.info(
        "Days post init in fitting data: " f"{my_data.n_days_post_init}"
    )
    logger.info(
        "First data date in fitting data: ",
        f"{my_data.first_data_date_overall.date()}",
    )
    forecast_data = my_data.to_forecast_data(n_forecast_points)
    logger.info(
        "Days post init in synthetic forecast data: "
        f"{forecast_data.n_days_post_init}"
    )
    logger.info(
        "First data date in synthetic forecast data: ",
        f"{forecast_data.first_data_date_overall.date()}",
    )
    logger.info(
        "Last data date in synthetic forecast data: ",
        f"{forecast_data.last_data_date_overall.date()}",
    )

    posterior_predictive = my_model.posterior_predictive(
        data=forecast_data,
        sample_ed_visits=True,
        sample_hospital_admissions=True,
        sample_wastewater=False,
    )

    idata = az.from_numpyro(
        my_model.mcmc, posterior_predictive=posterior_predictive
    )

    idata.to_dataframe().to_csv(model_dir / "inference_data.csv", index=False)

    # Save one netcdf for reloading
    idata.to_netcdf(model_dir / "inference_data.nc")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Do posterior prediction from a pyrenew-hew fit.")
    )
    parser.add_argument(
        "model_run_dir",
        type=Path,
        help=(
            "Path to a directory containing the model fitting data "
            "and the posterior chains. "
            "The completed predictive samples will be saved here."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use for generating predictions.",
    )
    parser.add_argument(
        "--n-forecast-points",
        type=int,
        default=0,
        help="Number of time points to forecast (Default: 0).",
    )
    args = parser.parse_args()

    generate_and_save_predictions(**vars(args))
