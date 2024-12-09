import argparse
import pickle
from pathlib import Path

import arviz as az
from build_model import build_model_from_dir


def generate_and_save_predictions(
    model_run_dir: str | Path, n_forecast_points: int
) -> None:
    model_run_dir = Path(model_run_dir)

    (
        my_model,
        data_observed_disease_ed_visits,
        right_truncation_offset,
    ) = build_model_from_dir(model_run_dir)

    my_model._init_model(1, 1)
    fresh_sampler = my_model.mcmc.sampler

    with open(
        model_run_dir / "posterior_samples.pickle",
        "rb",
    ) as file:
        my_model.mcmc = pickle.load(file)

    my_model.mcmc.sampler = fresh_sampler

    posterior_predictive = my_model.posterior_predictive(
        n_datapoints=len(data_observed_disease_ed_visits) + n_forecast_points
    )

    idata = az.from_numpyro(
        my_model.mcmc,
        posterior_predictive=posterior_predictive,
    )

    idata.to_dataframe().to_csv(
        model_run_dir / "inference_data.csv", index=False
    )

    # Save one netcdf for reloading
    idata.to_netcdf(model_run_dir / "inference_data.nc")

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
        "--n-forecast-points",
        type=int,
        default=0,
        help="Number of time points to forecast (Default: 0).",
    )
    args = parser.parse_args()

    generate_and_save_predictions(**vars(args))
