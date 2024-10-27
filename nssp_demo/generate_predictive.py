import argparse
import pickle
from pathlib import Path

import arviz as az
from build_model import build_model_from_dir


def generate_and_save_predictions(
        model_dir: str | Path,
        n_forecast_points: int
) -> None:

    model_dir = Path(model_dir)

    n_forecast_points = args.n_forecast_points
    (
        my_model,
        data_observed_disease_hospital_admissions,
        right_truncation_offset,
    ) = build_model_from_dir(model_dir)

    my_model._init_model(1, 1)
    fresh_sampler = my_model.mcmc.sampler

    with open(
            model_dir / "posterior_samples.pickle",
            "rb",
    ) as file:
        my_model.mcmc = pickle.load(file)
    
    my_model.mcmc.sampler = fresh_sampler

    posterior_predictive = my_model.posterior_predictive(
        n_datapoints=len(data_observed_disease_hospital_admissions)
        + n_forecast_points
    )

    idata = az.from_numpyro(
        my_model.mcmc,
        posterior_predictive=posterior_predictive,
    )

    idata.to_dataframe().to_csv(
        model_dir / "inference_data.csv", index=False)

    # Save one netcdf for reloading
    idata.to_netcdf(model_dir / "inference_data.nc")

    # R cannot read netcdf files with groups,
    # so we split them into separate files.
    for group in idata._groups_all:
        idata[group].to_netcdf(model_dir / f"inference_data_{group}.nc")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Do posterior prediction from pyrenew-hew"
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to the model directory containing the data.",
    )
    parser.add_argument(
        "--n_forecast_points",
        type=int,
        default=0,
        help="Number of time points to forecast",
    )
    args = parser.parse_args()

    generate_and_save_predictions(
        **vars(args))
