import argparse
import pickle
from pathlib import Path

import arviz as az
from jax.typing import ArrayLike

from pipelines.utils import get_priors_from_dir
from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData
from pyrenew_hew.utils import (
    build_pyrenew_hew_model,
    flags_from_pyrenew_model_name,
)


def generate_and_save_predictions(
    model_run_dir: str | Path,
    model_name: str,
    n_forecast_points: int,
    generation_interval_pmf: ArrayLike,
    inf_to_hosp_admit_lognormal_loc: ArrayLike,
    inf_to_hosp_admit_lognormal_scale: ArrayLike,
    inf_to_hosp_admit_pmf: ArrayLike,
    right_truncation_pmf: ArrayLike = None,
    predict_ed_visits: bool = False,
    predict_hospital_admissions: bool = False,
    predict_wastewater: bool = False,
) -> None:
    model_run_dir = Path(model_run_dir)
    model_dir = Path(model_run_dir, model_name)
    if not model_dir.exists():
        raise FileNotFoundError(f"The directory {model_dir} does not exist.")

    priors = get_priors_from_dir(model_run_dir)
    my_data = PyrenewHEWData.from_json(
        json_file_path=Path(model_run_dir)
        / "data"
        / "data_for_model_fit.json",
        **flags_from_pyrenew_model_name(model_name),
    )
    my_model = build_pyrenew_hew_model(
        priors,
        pop_fraction=my_data.pop_fraction,
        population_size=my_data.population_size,
        generation_interval_pmf=generation_interval_pmf,
        right_truncation_pmf=right_truncation_pmf,
        inf_to_hosp_admit_lognormal_loc=inf_to_hosp_admit_lognormal_loc,
        inf_to_hosp_admit_lognormal_scale=inf_to_hosp_admit_lognormal_scale,
        inf_to_hosp_admit_pmf=inf_to_hosp_admit_pmf,
        **flags_from_pyrenew_model_name(model_name),
    )

    my_model._init_model(1, 1)
    fresh_sampler = my_model.mcmc.sampler

    with open(
        model_dir / "posterior_samples.pickle",
        "rb",
    ) as file:
        my_model.mcmc = pickle.load(file)

    my_model.mcmc.sampler = fresh_sampler
    forecast_data = my_data.to_forecast_data(n_forecast_points)

    posterior_predictive = my_model.posterior_predictive(
        data=forecast_data,
        sample_ed_visits=predict_ed_visits,
        sample_hospital_admissions=predict_hospital_admissions,
        sample_wastewater=predict_wastewater,
    )

    idata = az.from_numpyro(
        my_model.mcmc, posterior_predictive=posterior_predictive
    )

    idata.to_dataframe().to_parquet(
        model_dir / "inference_data.parquet", index=False
    )

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
    parser.add_argument(
        "--predict-ed-visits",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="If provided, generate posterior predictions for ED visits.",
    )
    parser.add_argument(
        "--predict-hospital-admissions",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help=(
            "If provided, generate posterior predictions "
            "for hospital admissions."
        ),
    )
    parser.add_argument(
        "--predict-wastewater",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="If provided, generate posterior predictions for wastewater.",
    )

    args = parser.parse_args()

    generate_and_save_predictions(**vars(args))
